## Language Style Matching for Japanese（日本語版 README）

このリポジトリは、日本語テキストに対して **Language Style Matching (LSM)**、**reciprocal / directional LSM (rLSM)**、および **rolling-window rLSM (rw.rLSM)** を計算するための実装です。PhD 選考用の「研究実装の再現可能な成果物」として、定義（論文準拠）・入出力仕様・実行手順・テスト手順を README で明文化します。

- **英語版 README**: `README.md`（本ファイルは日本語版）
- **最小実行例**: `examples/`
- **サンプルデータ**: `sample_data/`
- **サンプル辞書（デモ用）**: `sample_dictionary/`

---

## Quick links（最短での動作確認）

### 前提（Dev Container を使う場合）

- Docker Desktop: `https://www.docker.com/products/docker-desktop`
- Visual Studio Code: `https://code.visualstudio.com/`
- Dev Containers extension: `https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers`

### まず動くことを確認（examples）

```bash
python examples/basic_lsm_example.py
python examples/basic_rlsm_example.py
```

補足:

- examples の説明（入出力形式・カスタマイズ）: `examples/README.md`
- サンプルデータの形式: `sample_data/README.md`
- サンプル辞書の注意（公式辞書は同梱不可など）: `sample_dictionary/README.md`

### テスト（再現性の確認）

```bash
pytest -v
```

---

## プロジェクト概要

- 目的: 会話における「内容語ではなくスタイル（機能語使用）」の同調（coordination / mimicry）を定量化する。
- 提供機能:
  - **LSM**: 会話全体（conversation-level）のスタイル一致度（方向性なし）
  - **rLSM**: 直前ターンに対する追従としての一致（方向性あり）
  - **rw.rLSM**: 移動窓（rolling window）で平滑化した rLSM（短ターンのノイズ緩和）
- 想定用途: 対話研究、医療面接/カウンセリング対話、協調作業対話などのスタイル同調分析（辞書ベースの手法として）

---

## 論文における定義（LSM / rLSM / rw.rLSM）

定義・数式の詳細は原典を参照してください。本実装は、原典の設計思想（特に rLSM の欠測処理）を日本語実装として再現できるようにしています。

### LSM（Language Style Matching：会話全体のスタイル一致）

- 定義: 二者（またはグループ）の対話において、**機能語（function words）カテゴリの使用率がどれだけ近いか**で言語スタイルの一致を定量化する。
- 計算単位: **会話全体（conversation-level）**。話者ごとにカテゴリ使用率を算出し、それらを比較して LSM を得る。
- スコア範囲: 設計上 **0–1**（高いほど一致）
- 性質: 方向性（誰が誰に合わせたか）は持たない

### rLSM（reciprocal / directional LSM：直前ターンへの追従としての一致）

- 定義: 会話全体の LSM では見えない時間的・話者間の非対称性を扱うため、**直前に話した相手の発話**に対して、**応答側がどれだけスタイルを合わせたか**をターン列で評価する。
- 計算要件: LIWC（または辞書ベース）指標を **各ターンごと**に算出する。
- 重要な性質: rLSM は **直前ターンに現れた機能語カテゴリ**を基準に一致度を計算する（追従可能性のあるカテゴリに限定して評価）。
- 欠測処理の重要性:
  - ターン単位ではカテゴリ出現がゼロになりやすく、「0」をそのまま扱うと一致度が歪む。
  - Müller-Frommeyer et al. (2019) は “0 を欠測として扱うべきケース”を明示し、適切な欠測処理の必要性を述べている。
  - 例：両者ともそのカテゴリを使っていない（0 と 0）の場合、0 を維持すると「完全一致」に見えるが、実際は“観測がない”だけである。
- 集約: カテゴリ別 rLSM を算出したのち、目的に応じて **話者単位**・**会話（ペア）単位**へ集約する。

### rw.rLSM（rolling-window rLSM：移動窓で平滑化した rLSM）

- 定義: rLSM は短いターンの影響で分散が大きくなりやすいため、複数ターンを束ねて平滑化し、**持続的なスタイル追従**を捉える目的で rolling-window 版を用いる。
- 代表的実装（論文例）:
  - 同一話者の隣接ターンを結合してから LIWC を適用（転記・ターン分割由来のノイズを軽減）
  - 各ターンに対して **同一話者の直前 7 ターンを加えた「8 ターン窓」**を作り、窓単位で LIWC→rLSM を適用して rw.rLSM を得る

---

## 日本語実装としての仕様（カテゴリ設計・前処理）

### A. 原典（英語 LIWC 9 カテゴリ）からの日本語カテゴリ設計

英語の 9 カテゴリに対し、日本語では J-LIWC のカテゴリ分けを踏まえて、以下の **7 カテゴリ**で LSM 系指標を算出します（本リポジトリのデフォルト設定）。

- Personal pronouns（代名詞）: `ppron`
- Indefinite pronouns（不定代名詞）: `ipron`
- Case particles（格助詞）: `casepart`
- Auxiliary verbs（助動詞）: `auxverb`
- Adverbs（副詞）: `adverb`
- Conjunctions（接続詞）: `conj`
- Negations（否定語）: `negate`

**設計方針**:

- 冠詞・数量詞は日本語に機能的対応が乏しいため除外
- 英語の前置詞は日本語の格助詞に対応づけ
- 英語の非人称代名詞は日本語の不定代名詞群に対応づけ
- J-LIWC は **カテゴリ同定目的**のみに用い、心理語用の解釈は行わない
- 言語差に応じた辞書カテゴリの適応は許容される（Müller-Frommeyer らの精神に沿う）

### B. 形態素解析・辞書照合（再現性の前提）

- 共通: 形態素解析は **spaCy + GiNZA（`ja_ginza`）** を使用し、lemma ベースで辞書照合します。
- rLSM / rw.rLSM: 環境によって `ja_ginza` が利用できない場合に `ja_ginza_electra` へフォールバックします（処理側で対応）。

---

## インストール手順

### Option 1: Dev Container（推奨）

VS Code + Dev Containers を使うと、同一環境を再現しやすく（再現性が高く）、ローカル環境を汚さずに実行できます。

**前提（Prerequisites）**:

- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [Visual Studio Code](https://code.visualstudio.com/)
- [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

**手順（Setup）**:

1. VS Code でこのリポジトリを開く
2. プロンプトが出たら `Reopen in Container`（またはコマンドパレットから `Dev Containers: Reopen in Container`）
3. 初回はコンテナビルドに数分かかります（目安: 5–10 分）
4. 依存関係は自動インストールされます

**Dev Container を推奨する理由**:

- ✅ クロスプラットフォームで環境が揃う（Windows/macOS/Linux）
- ✅ システム環境を汚さない（隔離環境）
- ✅ Python / spaCy / 依存が事前に揃った状態で実行できる
- ✅ この環境でテストが通ることを前提に検証しやすい
- ✅ 後片付けが簡単（コンテナを消せばよい）

**コンテナに含まれるもの（目安）**:

- Python 3.10
- `requirements.txt` の依存関係
- Git
- VS Code 拡張（例: Python, Pylance, Black, Ruff）
- pytest（テスト実行環境）

**セットアップの確認（Verifying the setup）**:

コンテナ内で以下を実行し、テストが通ることを確認してください。

```bash
pytest -v  # 31 passed が目安
```

**Troubleshooting**:

- **コンテナビルドが失敗する**: Docker Desktop が起動しているか、ディスク空き容量（目安: 2GB 以上）を確認してください
- **Docker が見つからない**: Docker Desktop のインストールと起動を確認してください
- **ビルドが遅い**: 初回はパッケージのビルドが走るため時間がかかることがあります（以降はキャッシュで高速化されます）
- **権限エラー**: コンテナは `vscode` ユーザーで動作し、権限問題を避ける設計です

**（代替）Docker CLI で直接実行する**:

VS Code を使わず Docker だけで動かす場合は以下を利用できます。

```bash
# Build the image
docker build -f .devcontainer/Dockerfile -t lsm .

# Run tests
docker run --rm -v "$(pwd):/workspace" lsm pytest -v

# Interactive shell
docker run --rm -it -v "$(pwd):/workspace" lsm bash
```

### Option 2: ローカル（venv）

**前提**:

- Python 3.8 以上
- pip

**手順**:

1. リポジトリを取得:

```bash
git clone <repository-url>
cd LanguageStyleMatching
```

2. 仮想環境を作成（推奨）:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

この手順で以下がインストールされます（例）:

- pandas, numpy（前処理）
- spacy, ja-ginza（日本語 NLP）
- pytest, pytest-cov, pytest-mock（テスト）

---

## 使用方法とサンプルコード

### Quickstart（まず動くことを確認）

```bash
python examples/basic_lsm_example.py
python examples/basic_rlsm_example.py
```

`examples/` は、この実装の最小動作例（入口）です。実データに適用する前に、ここで入出力の感覚と実行成功を確認してください。

### 入力データ形式（CSV）

会話データは CSV で、最低限以下を想定します（詳細は `examples/README.md` も参照）。

- `speaker`: 話者 ID（例: `female` / `male`、または `A` / `B` など）
- `text`: 発話（日本語テキスト）
- `start`:（任意）開始時刻（秒など）

### バッチ処理（研究用途向け）

複数ファイルをまとめて処理して CSV を出力する場合は、モジュール実行のランナーを利用します。

- LSM: `python -m lsm.runner --dic <dic_path> --results <out_dir> --round 1 2`
- rLSM: `python -m rlsm.runner --data <csv_dir> --dic <dic_path> --results <out_dir>`

※ オプションの詳細は各 `runner.py` の `--help` を参照してください。

---

## テスト実行方法

### 全テスト

```bash
pytest -v
```

### モジュール別

```bash
pytest lsm/tests/ -v
pytest rlsm/tests/ -v
```

### 特定のテストファイル

```bash
pytest lsm/tests/test_core.py -v
```

### テストカバレッジ（任意）

```bash
pytest --cov=lsm --cov=rlsm --cov-report=html
```

### 期待される結果

- 全テストは **31 件**が目安で、すべて PASS する想定です（LSM/rLSM コア、論文例の検証、runner/worker を含む）。

---

## 辞書（LIWC/J-LIWC）について

- `sample_dictionary/sample_liwc.dic` はデモ用の簡易辞書です。
- 実研究では、正式な辞書（例: J-LIWC 2015）を入手し、`--dic` やスクリプト内の辞書パスを差し替えてください。
- 辞書の取り扱い・注意点は `sample_dictionary/README.md` を参照してください。

---

## References（論文引用）

[1] Gonzales, Amy L., Jeffrey T. Hancock, and James W. Pennebaker. 2010. "Language Style Matching as a Predictor of Social Dynamics in Small Groups." Communication Research 37 (1): 3–19.

[2] Müller-Frommeyer, Lena C., Niels A. M. Frommeyer, and Simone Kauffeld. 2019. "Introducing rLSM: An Integrated Metric Assessing Temporal Reciprocity in Language Style Matching." Behavior Research Methods 51 (3): 1343–59.

[3] Khaleghzadegan, Salar, Michael Rosen, Anne Links, Alya Ahmad, Molly Kilcullen, Emily Boss, Mary Catherine Beach, and Somnath Saha. 2024. "Validating Computer-Generated Measures of Linguistic Style Matching and Accommodation in Patient-Clinician Communication." Patient Education and Counseling 119 (108074): 108074.

[4] Igarashi, Tasuku, Shimpei Okuda, and Kazutoshi Sasahara. 2022. "Development of the Japanese Version of the Linguistic Inquiry and Word Count Dictionary 2015." Frontiers in Psychology 13 (March): 841534.
