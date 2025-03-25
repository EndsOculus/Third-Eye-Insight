# Chat Interaction Affinity Analysis Tool

This project is a deep learning and natural language processing based tool for automated analysis of user interaction affinity in group and private chats. It extracts chat records from an unencrypted SQLite database, calculates interaction scores between users by combining text embeddings and behavioral statistics, and generates detailed visualizations and an automated analysis report.

## Project Differences & Extensions
Compared to the previous approach that simply extracted data from an unencrypted SQLite database, cleaned and normalized it, and then calculated interaction metrics based solely on weighted sums, this project introduces several enhancements:

- **Deep Learning Text Embedding:**  
  Instead of traditional text processing, this project uses deep learning to convert chat messages into embeddings, capturing semantic nuances.

- **Discrete Score Mapping:**  
  Both semantic similarity and behavior scores are mapped via an ECDF-based discrete function, providing a more granular and discrete distribution that highlights moderate differences while reducing extreme uniformity.

- **Fusion of Multiple Metrics:**  
  The final affinity score fuses both behavioral and semantic metrics, offering a more comprehensive view of user interactions.

- **User-Focused Analysis:**  
  With the `--focus-user` parameter, the analysis can be tailored to examine the interactions of a specific user within a group.

- **Enhanced Visualization and Reporting:**  
  The tool generates high-quality visualizations (heatmaps, network graphs) and can automatically produce a detailed analysis report using the DeepSeek API, providing insights beyond basic numerical outputs.

## Features

- **Data Extraction & Cleaning**  
  - Extracts chat data from SQLite databases (group_msg_table for groups and c2c_msg_table for private chats).  
  - Automatically filters out system messages (e.g., QQ numbers 10000 and 2854196310).  
  - Supports interactive time-range filtering (format: YYYY/MM/DD).

- **Interaction Metrics Calculation**  
  - Uses SentenceTransformer to obtain text embeddings and computes each user’s average embedding (mean pooling and normalization).  
  - Constructs a behavior matrix by counting consecutive interactions within a 5-minute window and applies an ECDF-based discrete mapping.  
  - Calculates a semantic similarity matrix (based on cosine similarity between average embeddings) and applies ECDF discrete mapping to amplify small differences.  
  - Fuses behavior and semantic scores (50% each) to yield the final affinity score.

- **Visualization & Report Generation**  
  - Generates heatmaps, network graphs, and other visualizations to depict user interactions.  
  - Optionally generates a detailed analysis report using the DeepSeek API.  
  - Supports focusing on a specific user (via the --focus-user parameter).

- **Flexible Command-line Parameters**  
  - Supports both group and private chat modes, specifying group/user IDs, time filtering, optional GPU acceleration, and report generation.

## Principle Explanation

The core principle of this project is divided into two parts:

1. **Text Embedding & Semantic Similarity**  
   - Each chat message is converted into a vector using a pre-trained SentenceTransformer.  
   - The average embedding for each user is computed (mean pooling followed by normalization) to represent their overall textual features.  
   - Cosine similarity between these average embeddings yields a semantic similarity matrix.  
   - An ECDF-based discrete mapping is applied to the similarity scores to spread out subtle differences across a discrete scale (0 to 1), emphasizing moderate similarities while reducing the frequency of extreme scores.

2. **Behavioral Statistics & Fusion**  
   - A behavior matrix is constructed by counting interactions (consecutive messages within a 5-minute window) between users.  
   - The interaction counts are log-transformed using np.log1p and then normalized via ECDF discrete mapping to obtain a behavior score.  
   - The final affinity score is the average of the behavior score and the mapped semantic similarity, ensuring that even in low-interaction scenarios the score is accurately represented.

## Implementation Method

The project is implemented through the following steps:

1. **Data Extraction & Cleaning**  
   - Extract chat records from the database based on the specified mode (group or private chat) and filter out system messages.  
   - Merge group nicknames and QQ names, and filter data by a user-specified time range.

2. **Text Feature Extraction**  
   - Use SentenceTransformer to compute text embeddings for each message.  
   - For each user, calculate the mean embedding (followed by normalization) to represent their overall textual profile.

3. **Semantic Similarity Calculation**  
   - Compute the cosine similarity matrix between users’ average embeddings.  
   - Apply an ECDF-based discrete mapping to the similarity scores to obtain a discrete distribution that spreads subtle differences.

4. **Behavioral Score Calculation**  
   - Count the number of consecutive interactions (within 5 minutes) between users to form a behavior matrix.  
   - Apply a log transformation and then normalize via ECDF discrete mapping.

5. **Score Fusion & Output**  
   - Fuse the discrete behavior score and semantic score (50% each) to compute the final affinity score.  
   - Save the results to a CSV file (GBK encoded), outputting unique user pairs.

6. **Visualization & Report Generation**  
   - Generate visualizations (heatmaps and network graphs) to display the interaction data.  
   - Optionally, call the DeepSeek API to generate a detailed Markdown analysis report.

## Example

Assume we have a group with group number **114514** and group name **Gensokyo**, and a user **1919810**. The example usernames are:  
- **Username 1: Marisa**  
- **Username 2: Reimu**

**Group Chat Mode Example:**

```bash
python train.py --group 114514 --db nt_msg.clean.db --mode group --id 114514 --focus-user 1919810 --font "Microsoft YaHei"
```

**Private Chat Mode Example:**

```bash
python train.py --group 0 --db nt_msg.clean.db --mode c2c --id 1919810 --font "Microsoft YaHei"
```

- **Lite Mode (`--lite`):**  
  When both the `--lite` and `--focus-user` parameters are specified, the program will filter the dataset to retain only those chat records that involve the specified focus user and interactions occurring within a 5-minute window. This allows for targeted analysis of the focus user’s interactions.  
  **Usage Example:**  
  ```bash
  python train.py --group 114514 --db nt_msg.clean.db --mode group --id 114514 --focus-user 1919810 --lite --font "Microsoft YaHei"
  ```

## Project Structure

```
QQ-Interaction-Analysis-Tool/
├── extract_chat_data.py       # Module for data extraction and cleaning
├── train.py                   # Main program: calculates and fuses interaction metrics, outputs CSV, charts, and report
├── visualization.py           # Module for generating visualizations (heatmaps, network graphs, etc.)
├── output/                    # Output directory: stores CSV, charts, and analysis report
└── README.md                  # This document
```

## Installation

Make sure you have Python 3.13.2 installed, then install the required dependencies via pip:

```bash
pip install pandas numpy matplotlib seaborn networkx torch sentence-transformers scipy requests scikit-learn
```

## Usage

Run `train.py` with the following command-line parameters:
- `--group`: Group chat number (used in group mode).
- `--db`: Path to the database file (e.g., `nt_msg.clean.db`).
- `--mode`: Mode; use `group` for group chat, `c2c` for private chat.
- `--id`: In group mode, same as `--group`; in c2c mode, the QQ number of the friend.
- `--focus-user`: Focus analysis on a specific user (QQ number).
- `--font`: Font for chart display (default: "Microsoft YaHei").
- `--boost`: Enable GPU acceleration (default off).
- `--report`: Generate a detailed analysis report (calls DeepSeek API).
- `--lite`：Keep only the interaction records related to a specific user to realize highly focused analysis.

Upon starting, the program will prompt you to enter a start date and an end date (format: YYYY/MM/DD):
- Enter both start and end dates to analyze data within that period.
- Press Enter without input to analyze all data.
- Only entering a start date means analyze from that date onward.
- Only entering an end date means analyze up to that date.

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
- [DeepSeek API](https://api.deepseek.com/)
