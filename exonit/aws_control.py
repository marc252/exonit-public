"""AWS EC2 control-plane helpers for ExoNit sector-2 (AWS) fleet management.

This module provides thin wrappers around boto3 EC2 to:

- Launch a Spot fleet of GPU workers via a Launch Template.
- Terminate all running/pending worker instances.
- Query fleet status (running/pending counts).
- Start, stop, and query the persistent AWS server instance.

The server instance MUST NEVER be terminated — it holds the EBS volume that
stores the SQLite database and the FITS/TGLC cache.  Only ``server_stop`` is
provided (not delete/terminate) to make accidental data loss hard.

Configuration is read from the environment on every call (not cached at
import time) so tests and deployment scripts can override values freely with
``os.environ`` / ``monkeypatch``:

- ``EXONIT_AWS_REGION``              — default ``"eu-west-3"``
- ``EXONIT_AWS_SERVER_INSTANCE_ID``  — default ``"i-00d0f6d5fa02663b4"``
- ``EXONIT_AWS_LAUNCH_TEMPLATE``     — default ``"exonit-worker"``

boto3 is imported lazily inside :func:`_ec2` so that importing this module
does not require boto3 to be installed — important for environments where
only mock-based tests run.
"""

from __future__ import annotations

import os
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_REGION = "eu-west-3"
_DEFAULT_SERVER_INSTANCE_ID = "i-00d0f6d5fa02663b4"
_DEFAULT_LAUNCH_TEMPLATE = "exonit-worker"

#: Tag applied to every worker instance so we can discover / terminate them.
WORKER_TAG: dict[str, str] = {"Key": "exonit-role", "Value": "worker"}

# States we consider "live" (need to be terminated when tearing down).
_LIVE_STATES = ["pending", "running"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _region() -> str:
    return os.environ.get("EXONIT_AWS_REGION", _DEFAULT_REGION)


def _server_id() -> str:
    return os.environ.get(
        "EXONIT_AWS_SERVER_INSTANCE_ID", _DEFAULT_SERVER_INSTANCE_ID
    )


def _launch_template_name() -> str:
    return os.environ.get("EXONIT_AWS_LAUNCH_TEMPLATE", _DEFAULT_LAUNCH_TEMPLATE)


def _ec2() -> Any:  # noqa: ANN401 — deliberately opaque; callers mock this
    """Return a boto3 EC2 client for the configured region.

    boto3 is imported here (lazily) so that importing ``exonit.aws_control``
    works even when boto3 is not installed — tests mock ``_ec2`` at the module
    level and never reach the real import.
    """
    import boto3  # noqa: PLC0415 — intentional lazy import

    return boto3.client("ec2", region_name=_region())


# ---------------------------------------------------------------------------
# Fleet management
# ---------------------------------------------------------------------------

def create_fleet(n: int, instance_type: str = "g6.xlarge") -> dict:
    """Launch *n* Spot worker instances via the configured Launch Template.

    Uses the EC2 Fleet API (``create_fleet``) with ``Type="maintain"`` so AWS
    keeps the fleet at the requested capacity even if individual instances are
    interrupted.

    Parameters
    ----------
    n:
        Total number of worker instances to launch.
    instance_type:
        EC2 instance type to request (e.g. ``"g6.xlarge"`` for L4 GPU).

    Returns
    -------
    dict
        The raw ``create_fleet`` response from boto3.
    """
    ec2 = _ec2()
    response = ec2.create_fleet(
        LaunchTemplateConfigs=[
            {
                "LaunchTemplateSpecification": {
                    "LaunchTemplateName": _launch_template_name(),
                    "Version": "$Latest",
                },
                "Overrides": [
                    {"InstanceType": instance_type},
                ],
            }
        ],
        TargetCapacitySpecification={
            "TotalTargetCapacity": n,
            "DefaultTargetCapacityType": "spot",
        },
        Type="maintain",
        # CreateFleet's TagSpecifications only accepts ResourceType "fleet"
        # (tagging the fleet request itself). Per-instance tags such as
        # WORKER_TAG must be declared in the Launch Template, where they are
        # applied to every instance the fleet launches. terminate_fleet() and
        # fleet_status() discover workers by that LT-applied instance tag.
        TagSpecifications=[
            {
                "ResourceType": "fleet",
                "Tags": [WORKER_TAG],
            }
        ],
    )
    return response


def terminate_fleet() -> dict:
    """Tear the worker fleet down completely.

    Two steps, in order:

    1. **Delete our EC2 Fleet requests.**  ``create_fleet`` uses
       ``Type="maintain"``, which *re-launches* replacement instances whenever
       they are terminated.  So terminating instances alone is not enough — the
       fleet request itself must be deleted or the workers come straight back.
       We delete every active fleet tagged with :data:`WORKER_TAG` (the tag
       ``create_fleet`` applies to the fleet request) with
       ``TerminateInstances=True``.
    2. **Terminate stragglers.**  Any remaining tagged instances not covered by
       a deleted fleet (e.g. left over from an earlier run) are terminated by
       the :data:`WORKER_TAG` instance tag.

    Returns
    -------
    dict
        ``{"terminated": [instance_id, ...], "deleted_fleets": [fleet_id, ...]}``.
    """
    ec2 = _ec2()

    # 1. Delete our maintain fleets so they stop replacing instances.
    fleets = ec2.describe_fleets().get("Fleets", [])
    our_fleets = []
    for f in fleets:
        if f.get("FleetState") not in ("active", "submitted", "modifying"):
            continue
        tags = {t.get("Key"): t.get("Value") for t in f.get("Tags", [])}
        if tags.get(WORKER_TAG["Key"]) == WORKER_TAG["Value"]:
            our_fleets.append(f["FleetId"])
    if our_fleets:
        ec2.delete_fleets(FleetIds=our_fleets, TerminateInstances=True)

    # 2. Terminate any straggler tagged instances.
    reservations = ec2.describe_instances(
        Filters=[
            {"Name": "tag:exonit-role", "Values": ["worker"]},
            {"Name": "instance-state-name", "Values": _LIVE_STATES},
        ]
    ).get("Reservations", [])

    ids = [
        inst["InstanceId"]
        for r in reservations
        for inst in r.get("Instances", [])
    ]

    if ids:
        ec2.terminate_instances(InstanceIds=ids)

    return {"terminated": ids, "deleted_fleets": our_fleets}


def fleet_status() -> dict:
    """Return counts of live worker instances grouped by state.

    Returns
    -------
    dict
        ``{"running": int, "pending": int, "instances": [instance_id, ...]}``.
    """
    ec2 = _ec2()
    reservations = ec2.describe_instances(
        Filters=[
            {"Name": "tag:exonit-role", "Values": ["worker"]},
            {"Name": "instance-state-name", "Values": _LIVE_STATES},
        ]
    ).get("Reservations", [])

    instances = [
        inst
        for r in reservations
        for inst in r.get("Instances", [])
    ]

    running = sum(
        1 for i in instances if i.get("State", {}).get("Name") == "running"
    )
    pending = sum(
        1 for i in instances if i.get("State", {}).get("Name") == "pending"
    )
    ids = [i["InstanceId"] for i in instances]

    return {"running": running, "pending": pending, "instances": ids}


# ---------------------------------------------------------------------------
# Server instance management
# ---------------------------------------------------------------------------
# IMPORTANT: The server instance MUST NEVER be terminated.  It holds the EBS
# volume containing the SQLite DB (exonit.db) and the FITS/TGLC cache.  Only
# start/stop is exposed here.  If the instance is accidentally terminated the
# EBS volume can still be recovered (it is not DeleteOnTermination), but
# rebuilding the environment is expensive.  Stop safely shuts down the OS and
# leaves the volume intact.

def server_start() -> dict:
    """Send a start-instances request for the server instance.

    Returns
    -------
    dict
        Raw boto3 ``start_instances`` response.
    """
    ec2 = _ec2()
    return ec2.start_instances(InstanceIds=[_server_id()])


def server_stop() -> dict:
    """Send a stop-instances request for the server instance.

    Stops the instance cleanly (OS shutdown) without terminating it so the
    EBS volume (DB + FITS cache) is preserved.  Never calls
    ``terminate_instances`` on the server.

    Returns
    -------
    dict
        Raw boto3 ``stop_instances`` response.
    """
    ec2 = _ec2()
    return ec2.stop_instances(InstanceIds=[_server_id()])


def server_status() -> dict:
    """Return the current state and public IP of the server instance.

    Returns
    -------
    dict
        ``{"state": str, "public_ip": str | None}``.
        ``public_ip`` is ``None`` when the instance is stopped or does not
        have a public IP assigned.
    """
    ec2 = _ec2()
    reservations = ec2.describe_instances(
        InstanceIds=[_server_id()]
    ).get("Reservations", [])

    instances = [
        inst
        for r in reservations
        for inst in r.get("Instances", [])
    ]

    if not instances:
        return {"state": "unknown", "public_ip": None}

    inst = instances[0]
    state = inst.get("State", {}).get("Name", "unknown")
    public_ip = inst.get("PublicIpAddress") or None
    return {"state": state, "public_ip": public_ip}
