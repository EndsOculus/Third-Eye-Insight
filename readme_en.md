# Chat Interaction Affinity Analysis Tool (Third Eye Insight)

This project is a deep learning and natural language processing based tool for automated analysis of user interaction affinity in group and private chats. It extracts chat records from an SQLite (plaintext or SQLCipher-encrypted) or remote PostgreSQL database, calculates interaction scores between users by combining text embeddings and behavioral statistics, and generates detailed visualizations and an automated analysis report.

## Features

- **Data Extraction & Cleaning**
  - Extracts chat data from SQLite databases (group_msg_table for groups, c2c_msg_table for private chats).
  - Automatically filters out system messages (QQ numbers 10000 and 2854196310).
  - Supports interactive time-range filtering (format: YYYY/MM/DD).

- **SQLCipher Encrypted Database Support**
  - Reads NTQQ's native encrypted database directly via `config.json` — no manual conversion needed.
  - Automatic detection of `nt_msg.db` at startup with a prompt to strip the 1024-byte header.

- **Interaction Metrics Calculation**
  - Uses SentenceTransformer to obtain text embeddings and computes each user's average embedding (mean pooling and normalization).
  - Constructs a behavior matrix by counting consecutive interactions within a 5-minute window.
  - Calculates a semantic similarity matrix (cosine similarity between average embeddings).
  - Applies ECDF-based discrete mapping to amplify subtle differences.
  - Fuses semantic, behavior, and network topology scores with configurable or auto-optimized weights.

- **Visualization & Report Generation**
  - Generates heatmaps (semantic, behavior, network), network graphs, and CSV output.
  - Optionally generates a detailed analysis report using the DeepSeek API.
  - Supports focusing on a specific user.

- **Interactive Configuration**
  - No command-line arguments — all options are presented as interactive prompts at startup.

## Principle

The core principle combines three metrics:

1. **Text Embedding & Semantic Similarity**
   Each message is converted to a vector using SentenceTransformer. Per-user average embeddings are computed and normalized; cosine similarity forms the semantic matrix. ECDF discrete mapping spreads scores across a discrete scale.

2. **Behavioral Statistics**
   Counts consecutive interactions within a 5-minute window. Log-transforms the counts and normalizes via ECDF mapping.

3. **Network Topology**
   Builds an interaction graph with NetworkX and computes degree centrality per user. The network score for a pair is the average of their centralities.

4. **Fusion**
   Three metrics are fused with configurable weights (default: semantic 0.4, behavior 0.4, network 0.2). Automatic weight optimization via scipy is also available.

## Project Structure

```
QQ-Interaction-Analysis-Tool/
├── extract_chat_data.py       # Data extraction and cleaning (SQLite / SQLCipher / PostgreSQL)
├── train.py                   # Main program: metrics calculation, fusion, output
├── visualization.py           # Heatmaps and network graph generation
├── config.json                # SQLCipher encryption configuration
├── output/                    # Output directory: CSV, charts, analysis report
└── README.md                  # This document
```

## Installation

Ensure you are using Python 3.13.2 (the official Windows installer, **not** the MSYS2 built-in Python).

**Using `uv` (recommended):**

> **Known issue — MSYS2 Python in PATH**: If MSYS2's Python (e.g. `C:/msys64/ucrt64/bin/python.exe`) appears before the Windows Python in your PATH, `uv pip install` will fail with `Unknown operating system: mingw_x86_64_ucrt_gnu`. Fix: specify the interpreter path explicitly.

> **Known issue — no `pyproject.toml`**: `uv add` requires a `pyproject.toml` and will fail with `No pyproject.toml found` in a plain script directory. Use `uv pip install` instead.

```bash
uv pip install --python "C:/Users/<your-username>/AppData/Local/Programs/Python/Python313/python.exe" \
  pandas numpy matplotlib seaborn networkx torch sentence-transformers scipy requests scikit-learn openai
```

**Or using pip directly:**

```bash
"C:/Users/<your-username>/AppData/Local/Programs/Python/Python313/python.exe" -m pip install \
  pandas numpy matplotlib seaborn networkx torch sentence-transformers scipy requests scikit-learn openai
```

For SQLCipher encrypted database support, install MSYS2 and run `pacman -S mingw-w64-x86_64-sqlcipher` so that `C:/msys64/mingw64/bin/libsqlcipher-0.dll` is available.

## Usage

Run the program and follow the interactive prompts:

```bash
uv run python train.py
```

The program will ask you step by step for:

| Prompt | Description |
|--------|-------------|
| Database path | Local SQLite path or PostgreSQL connection string; defaults to `db_file` in `config.json` |
| Analysis mode | 1 = group chat, 2 = private chat |
| Group / QQ number | Group ID or friend QQ number depending on mode |
| Focus user QQ number | Optional; leave blank to analyze all users |
| Lite mode | If a focus user is set, keep only that user's related records |
| Remote database | Use PostgreSQL connection string mode |
| GPU acceleration | Requires CUDA environment |
| Auto-adjust fusion weights | Uses scipy optimization |
| Generate AI report | Requires `DEEPSEEK_API_KEY` environment variable |
| Chart font | Default: Microsoft YaHei |

After data extraction, you will be prompted for an optional time range filter (format: YYYY/MM/DD).

## Database Preprocessing

**Encrypted NTQQ database (`nt_msg.db`):**
The program automatically detects `nt_msg.db` at startup and prompts you to strip the 1024-byte header to produce `nt_msg.clean_e.db`. You can also do this manually:

```bash
python -c "open('nt_msg.clean_e.db','wb').write(open('nt_msg.db','rb').read()[1024:])"
```

**`config.json` for SQLCipher decryption:**

```json
{
  "db_file": "nt_msg.clean_e.db",
  "encrypted": true,
  "sqlcipher_dll": "C:/msys64/mingw64/bin/libsqlcipher-0.dll",
  "cipher_page_size": 4096,
  "kdf_iter": 4000,
  "cipher_hmac_algorithm": "HMAC_SHA1",
  "cipher_kdf_algorithm": "PBKDF2_HMAC_SHA512",
  "password": "your_password_here"
}
```

Set `encrypted` to `false` (or omit it) for plaintext SQLite databases.

**PostgreSQL:** Ensure your connection string is correct and the service is running.

## Output

All output files are saved in the `output/` directory:

- **interaction_scores.csv**: CSV (GBK encoding) with unique user pair data: UserID, Nickname, BehaviorScore, SemanticScore, NetworkScore, Final Interaction Activity Score.
- **Heatmaps**: PNG files for semantic similarity, behavior scores, and network topology scores. File names include the time range if specified.
- **interaction_network.png**: Network graph of user interactions.
- **user_mapping.txt**: Maps user index, QQ number, and nickname.
- **analysis_report.md** (optional): Detailed analysis report generated via DeepSeek API.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [NetworkX](https://networkx.org/)
- [PyTorch](https://pytorch.org/)
- [SentenceTransformers](https://www.sbert.net/)
- [Scipy](https://www.scipy.org/)
- [DeepSeek API](https://api.deepseek.com/)
