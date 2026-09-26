## 0.0.28
+ Fixed Anthropic-compatible client rejecting `thinking` and `redacted_thinking` blocks replayed from prior assistant turns, which broke every multi-turn conversation with a thinking model.
+ Updated Anthropic-compatible client to reject `image`, `document`, and unrecognized content blocks with an `InvalidInput` error naming the block, instead of an opaque deserialization error.

## 0.0.27
+ Added `Parameter::context_length` field for the maximum context length in tokens of a chat message parameter.
+ Added `muna.beta.openai.models.retrieve` method, returning the OpenAI `Model` object for a compiled model.
+ Added `muna.beta.anthropic.models.retrieve` method, returning the Anthropic `ModelInfo` object for a compiled model.

## 0.0.26
+ Added support for Anthropic `thinking` and `output_config.effort` in `muna.beta.anthropic.messages.create`.

## 0.0.25
+ Added support for specifying `seed` in `muna.beta.openai.chat.completions.create`.
+ Added `ChatCompletionReasoningEffort::None` for disabling reasoning.
+ Fixed Anthropic-compatible client forwarding Claude Code's per-request `x-anthropic-billing-header` block to the model, which changed the prompt on every turn and defeated prefix caching.

## 0.0.24
+ Minor stability improvements.

## 0.0.23
+ Fixed OpenAI- and Anthropic-compatible clients rejecting assistant messages with empty `content`.
+ Fixed `null` assistant `content` being passed to predictors as `null` instead of `""`.
+ Fixed `MunaError::InvalidInput` messages carrying the Python exception class prefix (`ValueError: `, `TypeError: `) raised by the predictor.

## 0.0.22
+ Added request validation to ensure `messages` must be non-empty in OpenAI- and Anthropic-compatible clients.

## 0.0.21
+ Minor stability improvements.

## 0.0.20
+ Resource download improvements.

## 0.0.19
+ Minor updates.

## 0.0.18
+ Added support for tool calling in the OpenAI-compatible client.
+ Added support for tool calling in the Anthropic-compatible client.
+ Added support for OpenAI content parts in the OpenAI-compatible client. See `ChatCompletionContent` and `ChatCompletionContentPart` enums.
+ Added `MunaError::InvalidInput` error variant for caller-fault errors (e.g. unsupported or malformed content parts), allowing servers to surface them as client errors.
+ Refactored the Anthropic-compatible client into a pure adapter over the OpenAI chat completions contract. Chat predictors must yield `ChatCompletionChunk` outputs.

## 0.0.17
+ Added `muna.beta.anthropic.messages.create` method for creating messages with our Anthropic-compatible client.
+ Added `muna.beta.anthropic.messages.stream` method for streaming raw message stream events with our Anthropic-compatible client.
+ Refactored OpenAI-compatible client to require that chat predictors yield `ChatCompletionChunk` outputs. Predictors that return a full `ChatCompletion` are no longer supported.

## 0.0.16
+ Added `ChatCompletionMessage.reasoning_content` field in the OpenAI-compatible client, following the DeepSeek convention for reasoning models.
+ Added `ChatCompletionDelta.reasoning_content` field in the OpenAI-compatible client for streaming reasoning deltas.
+ Added `ChatCompletionUsage.completion_tokens_details` field in the OpenAI-compatible client for tracking reasoning tokens.

## 0.0.15
+ Minor improvements.

## 0.0.14
+ Added `muna.beta.openai.images.generate` method for generating images with our OpenAI-compatible client.
+ Added `BatchMode` type for inspecting the batching mode of a batched parameter.
+ Added `BatchConfig.mode` field to inspect parameter batching mode.
+ Added `ChatCompletionUsage.prompt_tokens_details` field with prompt token breakdowns in the OpenAI-compatible client.
+ Renamed `BatchConfig.max_count` field to `BatchConfig.capacity`.
+ Removed `Predictor.card` field.
+ Removed `Predictor.media` field.
+ Removed `Predictor.license` field.

## 0.0.13
+ Upgraded to Function C 0.0.48.

## 0.0.12
+ Improved prediction resource download speeds.

## 0.0.11
+ Minor stability improvements.

## 0.0.10
+ Added `muna.beta.openai.chat.completions.create` method for creating chat completions with our OpenAI-compatible client.
+ Added `muna.beta.openai.chat.completions.stream`  method for streaming chat completions with our OpenAI-compatible client.
+ Added support for remote and adaptive `acceleration` in `muna.preditions.create` method.
+ Removed `muna.beta.predictions.remote.create` method. Use `muna.predictions.create` method instead.
+ Removed `muna.beta.predictions.remote.stream` method. Use `muna.predictions.stream` method instead.
+ Removed `muna.beta.RemoteAcceleration` type. Use `muna.Acceleration` type instead.

## 0.0.9
+ Added support for making predictions with lists of tensors.
+ Added support for making predictions with lists of images.
+ Upgraded to Function C 0.0.43.

## 0.0.8
+ Added `Parameter.batch` field to inspect inference batching configuration for parameters.

## 0.0.7
+ Updated build script to export `DEP_FUNCTION_LIB_PATH` for path to the Function C library.

## 0.0.6
+ Updated build script to export `DEP_FUNCTION_LIB_DIR` for path to the Function C library.

## 0.0.5
+ Added `muna.beta.openai.embeddings.create` method for using text embedding models via an OpenAI-compatible client.

## 0.0.4
+ Fixed 403 error when making predictions that have not been cached on the local disk.

## 0.0.3
+ Added `Serialize` derive to `RemotePrediction` and `RemotePredictionEvent`.

## 0.0.2
+ Updated `muna.predictions.create` method to allow concurrent multi-threaded usage.

## 0.0.1
+ First pre-release.