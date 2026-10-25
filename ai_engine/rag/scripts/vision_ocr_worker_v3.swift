import Foundation
import Vision
import AppKit

var level: VNRequestTextRecognitionLevel = .accurate
var languageCorrection = true
var imagePaths: [String] = []

var i = 1
while i < CommandLine.arguments.count {
    let arg = CommandLine.arguments[i]
    if arg == "--level" && i + 1 < CommandLine.arguments.count {
        level = (CommandLine.arguments[i+1].lowercased() == "fast") ? .fast : .accurate
        i += 2
    } else if arg == "--correction" && i + 1 < CommandLine.arguments.count {
        languageCorrection = (CommandLine.arguments[i+1].lowercased() == "true")
        i += 2
    } else {
        imagePaths.append(arg)
        i += 1
    }
}

func processImage(path: String) -> [String: Any] {
    guard let img = NSImage(contentsOfFile: path),
          let cgImg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        return ["file": path, "lines": [], "line_count": 0, "status": "failed"]
    }
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = level
    req.usesLanguageCorrection = languageCorrection
    let handler = VNImageRequestHandler(cgImage: cgImg, options: [:])
    try? handler.perform([req])
    var lines: [String] = []
    if let obs = req.results {
        for o in obs {
            if let top = o.topCandidates(1).first?.string {
                lines.append(top)
            }
        }
    }
    return ["file": path, "lines": lines, "line_count": lines.count, "status": "ok"]
}

var results: [[String: Any]] = []
for p in imagePaths {
    results.append(processImage(path: p))
}

if let d = try? JSONSerialization.data(withJSONObject: results, options: []),
   let s = String(data: d, encoding: .utf8) {
    print(s)
} else {
    print("[]")
}
