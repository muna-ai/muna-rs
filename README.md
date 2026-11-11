# Muna for Rust
![Muna logo](https://raw.githubusercontent.com/muna-ai/.github/main/banner.png)

Run AI models in Rust.

___

## Useful Links
- [Check out several AI models we've compiled](https://github.com/muna-ai/muna-predictors).
- [Join our Slack community](https://muna.ai/slack).
- [Check out our docs](https://docs.muna.ai).
- [Read our blog](https://muna.ai/blog).
- Reach out to us at [hi@muna.ai](mailto:hi@muna.ai).

Muna is a product of [NatML Inc](https://github.com/natmlx).

### Encoded image prediction transport

`RemoteValue.data` accepts either an encoded URL string or an array of
`RemoteValue` children. Composed `image_list` values contain `image` leaves;
other compositions, nested lists, and lists longer than 1024 are rejected.
Existing string payloads, including AVIF sequences, remain supported.

```json
{"dtype":"image_list","data":[{"dtype":"image","data":"data:image/jpeg;base64,..."}]}
```

The OpenAI images adapter preserves encoded input files for remote execution.
The control plane configures its request-scoped `NodeClient` to request response
encoding with these headers, without adding fields to the prediction body:

```http
X-Muna-Image-Output-Format: jpeg
X-Muna-Image-Output-Quality: 90
```

Formats are `png`, `jpeg`, and `webp`; optional quality is an integer from 0 to
100 and requires a format. Quality affects JPEG/WebP; PNG remains lossless.
The node encodes each image once and returns composed image values. The
OpenAI adapter passes their base64 data through without decoding. Prediction
IDs, timestamps, latency, errors, and other outputs remain in the prediction
envelope for usage accounting. Headers apply to all image outputs, including
streamed predictions. Without headers, output serialization retains its legacy
behavior. The SDK's `raw` output continues to return decoded pixels. The generic SDK
prediction service has no encoding options or header policy. Without an
application wrapper, the image adapter accepts legacy AVIF responses and
converts them to the requested format.

Roll out node support before enabling the new SDK/control-plane path. The
server and control plane require muna-rs 0.0.38; publish that SDK before release
builds, then run `cargo update -p muna --precise 0.0.38` in the server and
control-plane repositories to refresh their registry lock entries. For local validation against the sibling checkout, use
`--config 'patch.crates-io.muna.path="../muna-rs"'` with an absolute path if needed.
In Rust, existing constructors supplying `RemoteValue.data: Some(string)` must
now use `Some(string.into())`; the OpenAI images public method signatures are
unchanged.
