# Legal DMS

Legal DMS is a local-first document management scaffold for legal teams, designed to grow into a system that combines OCR, LLM classification, document organization, vector indexing, and an elegant Streamlit interface.

## Architecture

- **OCR**: Extracts text from scanned documents and images, producing searchable content for later classification and indexing.
- **Classifier**: Labels documents using an LLM-based classification layer, enriching metadata and enabling automated routing.
- **Organizer**: Applies business rules and folder organization logic to move documents from an inbox into a structured library and working area.
- **Indexer**: Builds a vector-enabled search index from processed documents so semantic search and retrieval work efficiently.
- **Watcher**: Observes local folders for new files and triggers the pipeline automatically in a local-first workflow.
- **Pipeline**: Connects the OCR, Classifier, Organizer, and Indexer steps into a repeatable ingestion flow.
- **Chat**: Provides an LLM-assisted conversational access layer to search, summarize, and retrieve documents.
- **UI**: Hosts a minimalist Streamlit front-end that uses shared design tokens from the core package to ensure a consistent visual identity.

## How to run

Install dependencies with `uv` if available, or fall back to `pip`.

```powershell
cd "C:\Ai Documanager\legal-dms"
uv install
```

If `uv` is unavailable:

```powershell
python -m pip install --upgrade pip
python -m pip install uv pydantic-settings streamlit python-dotenv
```

Run the Streamlit app:

```powershell
uv run streamlit run src/legal_dms/ui/app.py
```

If `uv` is unavailable, use:

```powershell
python -m streamlit run src/legal_dms/ui/app.py
```
