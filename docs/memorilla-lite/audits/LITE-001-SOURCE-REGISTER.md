# LITE-001 Source and Citation Register

## Resolution policy

The imported report contains opaque citation identifiers from its research environment. No mapping is invented. A token is marked resolved only when a canonical public source is directly available and inspected. The configured web-search provider was unavailable because `BRAVE_SEARCH_API_KEY` is not set, so report-internal tokens remain unresolved in this audit.

## Directly inspected local sources

| Source | Evidence | Status |
|---|---|---|
| Upstream `README.md` | Memorilla architecture, Qwen3/H100 requirements, installation/training/evaluation, benchmark claims | inspected at baseline commit `3ce5497` |
| Upstream `LICENSE` | Apache License 2.0 text | inspected; preserved |
| Imported `research/memorilla.txt` | Newport River research report | byte/hash verified |
| Android runtime commands | Device identity, memory, storage and capability probes | measured in this audit |

## Unresolved report identifiers

The following 36 opaque identifiers remain unresolved and must not be treated as canonical citations without later source recovery:

```text
turn137545view0
turn224823view0
turn224823view1
turn224823view2
turn236805view0
turn379802search0
turn379802search14
turn379802search16
turn379802search20
turn379802search3
turn379802search33
turn379802search6
turn417960search0
turn417960search11
turn417960search12
turn417960search18
turn507950search2
turn615191search1
turn615191search2
turn615191search3
turn615191search7
turn774609view0
turn774609view1
turn856967search0
turn856967search1
turn856967search2
turn856967search3
turn931124search1
turn969170search0
turn969170search15
turn969170search2
turn969170search37
turn969170search4
turn980912view0
turn980912view1
turn980912view2
```

## Runtime selection conclusion

ONNX Runtime remains the recommended first architecture from the research baseline, but it is not installed or verified on this device. Native Termux is the least invasive prototype path if a compatible prebuilt ARM64 package is available. An Android-native/C++ layer is the fallback if native Termux packaging cannot provide a compatible runtime. ExecuTorch and ncnn remain later alternatives. `llama.cpp` is deferred because no local binary/library/model exists and decoder feasibility was not part of this read-only audit.
