//! Minimal single-threaded HTTP/1.1 test server, standing in for the Swift
//! contract suite's `MockURLProtocol` (which intercepts `URLSession`
//! in-process). Rust's `ureq` has no equivalent request-interception hook,
//! and no HTTP-mocking crate is in the F0b dependency allowlist (`ureq`,
//! `rusqlite` only — see docs/DESIGN.md), so this listens on a real loopback
//! TCP port instead.
//!
//! Deliberately minimal: one request per connection (`Connection: close` is
//! always sent, and the client is expected to open a fresh connection per
//! request — `ureq`'s default `Agent` does this transparently), no
//! keep-alive, no chunked transfer-encoding parsing. That is enough to speak
//! to `ureq`'s client requests, which is the only client that will ever hit
//! this server.
//!
//! The background accept thread is never explicitly joined/stopped — it
//! blocks in `accept()` for the lifetime of the test binary process, which
//! is fine (and standard practice for this kind of test helper): each test
//! binds an ephemeral port (`127.0.0.1:0`), so tests never collide, and the
//! OS reclaims the thread/socket when the test process exits.
//!
//! This module is included via `#[path = "support/mod.rs"]` into several
//! independent test binaries (`api_tests.rs`, `outbox_tests.rs`,
//! `runtime_tests.rs`, ...); each binary only exercises the subset of this
//! API it needs, so `#![allow(dead_code)]` avoids spurious per-binary
//! "never used" warnings on the rest.
#![allow(dead_code)]

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};

#[derive(Debug, Clone)]
pub struct MockRequest {
    pub method: String,
    pub path: String,
    /// Header names are stored lower-cased for case-insensitive lookup via
    /// `header()`, matching HTTP's case-insensitive header name semantics.
    headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

impl MockRequest {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(|s| s.as_str())
    }

    pub fn body_json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).expect("mock request body is valid JSON")
    }
}

pub struct MockResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl MockResponse {
    pub fn json(status: u16, body: &str) -> Self {
        MockResponse {
            status,
            headers: vec![("Content-Type".to_string(), "application/json".to_string())],
            body: body.as_bytes().to_vec(),
        }
    }

    pub fn bytes(status: u16, content_type: &str, body: Vec<u8>) -> Self {
        MockResponse {
            status,
            headers: vec![("Content-Type".to_string(), content_type.to_string())],
            body,
        }
    }

    pub fn with_header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.to_string(), value.to_string()));
        self
    }
}

pub struct MockServer {
    pub base_url: String,
}

impl MockServer {
    /// Starts the server on an ephemeral loopback port, dispatching every
    /// request to `handler` (called once per request, from the background
    /// accept thread — tests share state into it via `Arc`/`Mutex`, mirroring
    /// how the Swift tests close over a `LockedCounter`).
    pub fn start<F>(handler: F) -> Self
    where
        F: Fn(&MockRequest) -> MockResponse + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        let port = listener
            .local_addr()
            .expect("mock server local addr")
            .port();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                handle_connection(stream, &handler);
            }
        });
        MockServer {
            base_url: format!("http://127.0.0.1:{port}"),
        }
    }
}

fn handle_connection<F>(stream: TcpStream, handler: &F)
where
    F: Fn(&MockRequest) -> MockResponse,
{
    let mut reader = BufReader::new(stream.try_clone().expect("clone mock server stream"));
    let Some(request) = read_request(&mut reader) else {
        return;
    };
    let response = handler(&request);
    write_response(stream, &response);
}

fn read_request(reader: &mut BufReader<TcpStream>) -> Option<MockRequest> {
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).ok()? == 0 {
        return None;
    }
    let mut parts = request_line.trim_end().split(' ');
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();

    let mut headers = HashMap::new();
    let mut content_length: usize = 0;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).ok()? == 0 {
            break;
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        let (name, value) = trimmed.split_once(':')?;
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim().to_string();
        if name == "content-length" {
            content_length = value.parse().ok()?;
        }
        headers.insert(name, value);
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body).ok()?;
    }
    Some(MockRequest {
        method,
        path,
        headers,
        body,
    })
}

fn write_response(mut stream: TcpStream, response: &MockResponse) {
    let reason = reason_phrase(response.status);
    let mut head = format!("HTTP/1.1 {} {}\r\n", response.status, reason);
    for (name, value) in &response.headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str(&format!("Content-Length: {}\r\n", response.body.len()));
    head.push_str("Connection: close\r\n\r\n");
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(&response.body);
    let _ = stream.flush();
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        202 => "Accepted",
        401 => "Unauthorized",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Unknown",
    }
}
