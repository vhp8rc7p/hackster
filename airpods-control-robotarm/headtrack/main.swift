import Foundation
import CoreMotion
import Darwin

// AirPods head pose -> UDP JSON on 127.0.0.1:9870, ~25 Hz.
// Fan out to every consumer: 9870 = arm bridge, 9871 = visualizer.
// Independent sockets so either can be absent without affecting the other.
let PORTS: [UInt16] = [9870, 9871]
let verbose = CommandLine.arguments.contains("--print")

let fd = socket(AF_INET, SOCK_DGRAM, 0)
var targets: [sockaddr_in] = PORTS.map { port in
    var a = sockaddr_in()
    a.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    a.sin_family = sa_family_t(AF_INET)
    a.sin_port = port.bigEndian
    a.sin_addr.s_addr = inet_addr("127.0.0.1")
    return a
}

func emit(_ s: String) {
    let bytes = Array(s.utf8)
    for i in targets.indices {
        withUnsafePointer(to: &targets[i]) { p in
            p.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                _ = sendto(fd, bytes, bytes.count, 0, sa, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
    }
}

func log(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

let mgr = CMHeadphoneMotionManager()
log("auth=\(CMHeadphoneMotionManager.authorizationStatus().rawValue) available=\(mgr.isDeviceMotionAvailable)")

guard mgr.isDeviceMotionAvailable else { log("CoreMotion headphone API unavailable"); exit(1) }

var n = 0
let t0 = Date()
let deg = 180.0 / Double.pi

mgr.startDeviceMotionUpdates(to: .main) { motion, error in
    if let error { log("error: \(error.localizedDescription)"); return }
    guard let m = motion else { return }
    let a = m.attitude
    n += 1
    let q = a.quaternion
    emit(String(format:
        "{\"t\":%.4f,\"pitch\":%.4f,\"roll\":%.4f,\"yaw\":%.4f," +
        "\"qx\":%.5f,\"qy\":%.5f,\"qz\":%.5f,\"qw\":%.5f," +
        "\"wx\":%.4f,\"wy\":%.4f,\"wz\":%.4f,\"ax\":%.4f,\"ay\":%.4f,\"az\":%.4f}",
        m.timestamp, a.pitch * deg, a.roll * deg, a.yaw * deg,
        q.x, q.y, q.z, q.w,
        m.rotationRate.x, m.rotationRate.y, m.rotationRate.z,
        m.userAcceleration.x, m.userAcceleration.y, m.userAcceleration.z))
    if verbose || n % 25 == 0 {
        let hz = Double(n) / max(Date().timeIntervalSince(t0), 0.001)
        log(String(format: "#%-5d %4.1fHz  pitch %+7.2f  roll %+7.2f  yaw %+7.2f",
                   n, hz, a.pitch * deg, a.roll * deg, a.yaw * deg))
    }
}

signal(SIGINT) { _ in exit(0) }
log("streaming to udp://127.0.0.1:\(PORTS.map(String.init).joined(separator: ",")) — Ctrl-C to stop")
RunLoop.main.run()
