# Corpus sources and licensing

## Current corpus: English Wikipedia (legal articles)

- **Source:** English Wikipedia articles listed in `lawbot/data/sources.py`.
- **Retrieved via:** the public MediaWiki API (`action=query&prop=extracts|info`).
- **License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- **Attribution:** every indexed chunk stores `title`, `source`, `url`, and `license` in its
  LangChain `Document.metadata`. The RAG prompt requires numbered citations, and the API
  returns a `sources` array with titles, URLs, and licenses for every answer.
- **Caching:** raw extracts are written to `data/raw/wikipedia_legal.json` (gitignored) so
  the index can be rebuilt without re-hitting Wikipedia. Re-fetch with `--refresh`.
- **Politeness:** requests are batched (10 titles per call) with a short pause and a
  descriptive User-Agent.

## Why this corpus

Wikipedia is a genuine *encyclopedia* with broad conceptual coverage of legal topics and a
clear, redistributable license. Many commercial legal encyclopedias (for example
Black's Law Dictionary, Westlaw, Britannica) are copyrighted and must not be scraped into
a corpus without a license, so they are deliberately excluded.

## Planned extension: primary law (public domain)

The next corpus layer can add public-domain primary sources such as:

- US Code / CFR / federal court opinions (US government works are public domain),
- `legislation.gov.uk` material under the Open Government Licence,
- EUR-Lex documents subject to the EU reuse notice.

Each new source must be added here with its exact license and attribution requirements
before ingestion. Use the `source` and `license` metadata fields to keep citations accurate.

## Disclaimer

The assistant provides general legal information, not legal advice. Responses are limited
to retrieved passages and always carry this disclaimer.
