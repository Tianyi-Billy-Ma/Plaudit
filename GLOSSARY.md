# Plaudit

Meeting recordings from any device become speaker-labelled transcripts and minutes on hardware the user controls.

## Language

**Recording**:
One audio file of one meeting, in whatever format the device produced.
_Avoid_: file, clip (a clip is a slice of a recording)

**Source**:
A place recordings arrive from: the watched folder, an rclone remote, Plaud, the upload endpoint.
_Avoid_: input, provider

**Job**:
One recording's passage through the queue, with its status, attempts and links.
_Avoid_: task, run

**Queue**:
The ordered set of jobs on one data directory, processed one at a time.
_Avoid_: daemon (the process that drives the queue), scheduler

**Turn**:
A stretch of one speaker's speech found by diarization, with its start and end.
_Avoid_: segment (the code's name for the data shape), utterance

**Transcript**:
All turns of a recording with their text filled in.
_Avoid_: ASR output

**Glossary** (the user's):
Terms the user wants recognised as spoken, given to transcription and summarization.
_Avoid_: vocabulary, hot words

**Minutes**:
The summary of one meeting: summary, decisions, action items.
_Avoid_: notes (the Markdown file that holds minutes plus transcript), summary (one section of the minutes)

**Chat adapter**:
Whatever answers one system-plus-user prompt with text for the summarizer; the OpenAI-compatible endpoint in production, a fake in tests.
_Avoid_: LLM client, backend

**Sink**:
A place minutes land: Markdown folder, Notion database, webhook, notification.
_Avoid_: output (the config section's name), destination, integration

**Stuck**:
A job that has kept failing for a day; it gets a stub page in the sinks that support one.
_Avoid_: failed, dead
