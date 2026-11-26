use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::UdpSocket;

use crate::network::servers::{get_local_ip, ping_server, ServerEntry};

pub const DEFAULT_LAN_PORT: u16 = 8090;
pub const LAN_DISCOVERY_PORT: u16 = 30009;
pub const LAN_MULTICAST_IPV4: Ipv4Addr = Ipv4Addr::new(239, 255, 42, 99);
pub const LAN_MAGIC_PREFIX: &str = "CBX_LAN_V1:";
pub const LAN_PROBE_MSG: &[u8] = b"CBX_DISCOVER_V1";

pub fn local_lan_ipv4() -> String {
    get_local_ip()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanBeaconInfo {
    pub name: String,
    pub port: u16,
    pub motd: String,
    pub players_online: u32,
    pub players_max: u32,
}

impl LanBeaconInfo {
    pub fn encode_packet(&self) -> Vec<u8> {
        let json = serde_json::to_string(self).unwrap_or_default();
        format!("{LAN_MAGIC_PREFIX}{json}").into_bytes()
    }

    pub fn decode_packet(data: &[u8]) -> Option<Self> {
        let s = std::str::from_utf8(data).ok()?;
        let rest = s.strip_prefix(LAN_MAGIC_PREFIX)?;
        serde_json::from_str::<Self>(rest).ok()
    }
}

/// Active LAN UDP broadcast + multicast responder for hosted Cubex worlds.
pub struct LanBroadcaster {
    stop_flag: Arc<AtomicBool>,
}

impl LanBroadcaster {
    pub fn spawn(info: LanBeaconInfo) -> Self {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_c = stop_flag.clone();
        tokio::spawn(async move {
            let payload = info.encode_packet();
            let bcast_addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::BROADCAST, LAN_DISCOVERY_PORT));
            let mcast_addr = SocketAddr::V4(SocketAddrV4::new(LAN_MULTICAST_IPV4, LAN_DISCOVERY_PORT));

            // Optional listener socket for immediate replies to CBX_DISCOVER_V1
            let listener = UdpSocket::bind(("0.0.0.0", LAN_DISCOVERY_PORT)).await.ok();
            if let Some(ref sock) = listener {
                let _ = sock.set_broadcast(true);
                let _ = sock.join_multicast_v4(LAN_MULTICAST_IPV4, Ipv4Addr::UNSPECIFIED);
            }
            let sender = UdpSocket::bind("0.0.0.0:0").await.ok();
            if let Some(ref sock) = sender {
                let _ = sock.set_broadcast(true);
            }

            let mut buf = [0u8; 512];
            let mut next_announce = Instant::now();
            while !stop_c.load(Ordering::Relaxed) {
                if Instant::now() >= next_announce {
                    if let Some(ref sock) = sender {
                        let _ = sock.send_to(&payload, bcast_addr).await;
                        let _ = sock.send_to(&payload, mcast_addr).await;
                    }
                    next_announce = Instant::now() + Duration::from_millis(1500);
                }

                if let Some(ref sock) = listener {
                    if let Ok(Ok((n, src))) =
                        tokio::time::timeout(Duration::from_millis(200), sock.recv_from(&mut buf)).await
                    {
                        if &buf[..n] == LAN_PROBE_MSG {
                            let _ = sock.send_to(&payload, src).await;
                        }
                    }
                } else {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            }
        });
        Self { stop_flag }
    }

    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::Relaxed);
    }
}

impl Drop for LanBroadcaster {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Send UDP broadcast + multicast probes and listen for LAN beacon announcements.
pub async fn discover_lan_udp(timeout: Duration) -> Vec<ServerEntry> {
    let mut found = Vec::new();
    let mut seen: HashSet<(String, u16)> = HashSet::new();

    let Ok(sock) = UdpSocket::bind("0.0.0.0:0").await else {
        return found;
    };
    let _ = sock.set_broadcast(true);
    let _ = sock.join_multicast_v4(LAN_MULTICAST_IPV4, Ipv4Addr::UNSPECIFIED);

    let bcast = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::BROADCAST, LAN_DISCOVERY_PORT));
    let mcast = SocketAddr::V4(SocketAddrV4::new(LAN_MULTICAST_IPV4, LAN_DISCOVERY_PORT));
    let local_loop = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, LAN_DISCOVERY_PORT));

    let t0 = Instant::now();
    let _ = sock.send_to(LAN_PROBE_MSG, bcast).await;
    let _ = sock.send_to(LAN_PROBE_MSG, mcast).await;
    let _ = sock.send_to(LAN_PROBE_MSG, local_loop).await;

    let deadline = t0 + timeout;
    let mut buf = [0u8; 1024];
    while Instant::now() < deadline {
        let rem = deadline.saturating_duration_since(Instant::now());
        match tokio::time::timeout(rem, sock.recv_from(&mut buf)).await {
            Ok(Ok((n, src))) => {
                if let Some(info) = LanBeaconInfo::decode_packet(&buf[..n]) {
                    let host = src.ip().to_string();
                    if seen.insert((host.clone(), info.port)) {
                        let ping_ms = t0.elapsed().as_millis().max(1) as u32;
                        found.push(ServerEntry {
                            name: info.name,
                            host,
                            port: info.port,
                            motd: info.motd,
                            featured: false,
                            category: "LAN".into(),
                            players_online: info.players_online,
                            players_max: info.players_max,
                            ping_ms: Some(ping_ms),
                            online: true,
                            is_lan: true,
                        });
                    }
                }
            }
            _ => break,
        }
    }
    found
}

/// Combined LAN discovery: UDP broadcast/multicast beacons + localhost/LAN port probes.
pub async fn discover_lan_servers(timeout: Duration) -> Vec<ServerEntry> {
    let mut results = discover_lan_udp(timeout).await;
    let mut seen: HashSet<(String, u16)> = results.iter().map(|s| (s.host.clone(), s.port)).collect();

    let mut hosts = vec!["127.0.0.1".to_string()];
    let local_ip = get_local_ip();
    if !local_ip.is_empty() && local_ip != "127.0.0.1" {
        hosts.push(local_ip);
    }

    let mut tasks = tokio::task::JoinSet::new();
    for host in hosts {
        for port in [DEFAULT_LAN_PORT, DEFAULT_LAN_PORT + 1, 30000, 30001] {
            let h = host.clone();
            tasks.spawn(async move {
                let (online, ping_ms) = ping_server(&h, port, Duration::from_millis(180)).await;
                (h, port, online, ping_ms)
            });
        }
    }

    while let Some(Ok((host, port, online, ping_ms))) = tasks.join_next().await {
        if online && seen.insert((host.clone(), port)) {
            results.push(ServerEntry {
                name: format!("LAN World ({host}:{port})"),
                host,
                port,
                motd: "§aLocal Cubex World\n§7Click Join Server to connect!".into(),
                featured: false,
                category: "LAN".into(),
                players_online: 1,
                players_max: 16,
                ping_ms,
                online: true,
                is_lan: true,
            });
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lan_beacon_packet_roundtrip() {
        let info = LanBeaconInfo {
            name: "Local Test Realm".into(),
            port: 30001,
            motd: "§aTesting LAN".into(),
            players_online: 2,
            players_max: 16,
        };
        let pkt = info.encode_packet();
        let dec = LanBeaconInfo::decode_packet(&pkt).expect("decode lan packet");
        assert_eq!(dec.name, "Local Test Realm");
        assert_eq!(dec.port, 30001);
        assert_eq!(dec.players_online, 2);
    }
}
