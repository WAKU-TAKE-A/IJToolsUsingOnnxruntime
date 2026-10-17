# 開発履歴 - IJToolsUsingOnnxruntime

## [0.9.5] - 2026-10-17

### 追加
- 共通クラス `Nms` に NMS を集約。RF-DETR の3タスクにもクラス別 NMS を追加し、`nms_threshold` を有効化。
- Roboflow RF-DETR の物体検出、インスタンスセグメンテーション、姿勢推定（Keypoint Preview）に対応。
- 既存のモデル管理に `RF_DETR_Object` / `RF_DETR_Segment` / `RF_DETR_Pose` を追加。
- RF-DETR 専用の前処理、global top-K、マスク補間、キーポイントの uncertainty fusion と、その後の追加 NMS を実装。
- `RF-DETR/` に仮想環境セットアップ、Apache-2.0 モデルの取得・ONNX 変換、Java と公式 Python の比較スクリプトを追加。
- サンプル画像による3タスクの実推論と ImageJ マクロを検証。モデル・サンプル・検証画像は `RF-DETR/models/` に保存。

### 修正
- 配布 JAR に本プロジェクトと ONNX Runtime の MIT ライセンス全文・同梱内容の案内を追加。
- モデル読み込みダイアログの確定値を明示的に取得し、マクロで追加形式を選択できるようにした。
- 推論入力テンソルとセッション設定を try-with-resources で解放するようにした。
- YOLO の仮想環境セットアップを、起動時のカレントディレクトリに依存しないよう修正。

### ドキュメント
- RF-DETR と YOLO のスクリプトの実行方法・制約を各 README に整理。
- エージェント向けのコード地図 `README_AI.md` を追加。

## [0.9.2] - 2026-05-07

### 追加
- 全プラグイン（Detection, Pose, Classify）に `enable_refresh_data` チェックボックスを追加。結果テーブルや ROI マネージャーの「クリアして追記」か「単純追記」かを選択可能に。

### 修正
- YOLOX 向けの前処理を適正化。BGR 順かつ [0, 255] スケールでの入力をサポート。
- YOLOX Undecoded 形式のデコード処理を修正。グリッドとストライドを用いた座標変換ロジックを実装。
- Headless モードおよびマクロ実行時の引数解析不具合を修正。`showDialog` 内でのパラメータ取得を確実に行うよう改善。
- `OrtUtil` の `ResultsTable` および `RoiManager` 取得ロジックを修正し、Headless/マクロ環境でのデータ共有を安定化。

## [0.9.1] - 2026-05-07

### 修正
- YOLOX Undecoded 形式のデコード処理を修正。グリッドとストライドを用いた座標変換ロジックを追加し、モデル出力の解析精度を改善。

## [0.9.0] - 2026-05-06

### 変更・改善
- エラーハンドリングの刷新：`IJ.error` を廃止し、共通ログ関数 `OrtUtil.logError` によるログ出力へ移行。
- 設定ダイアログの改善：`dialogItemChanged` でのエラー通知をステータスバー（`IJ.showStatus`）に変更し、ユーザー体験を向上。
- モデル管理の改善：既存スロットへのロード時の上書き確認を削除し、強制上書きに変更。

## [0.1.0] - 2026-05-05

### 新規追加
- ONNX Runtime を使用した ImageJ プラグイン群の初期実装。
- **ORT_1st_ReadAndRelease**: スロットベースのモデル管理（読み込み・解放）。
- **ORT_2nd_Detection**: YOLO 物体検出（通常モデル・E2E モデル対応）。
- **ORT_2nd_Pose**: YOLO ポーズ推定（通常モデル・E2E モデル対応）。
- **ORT_2nd_Classify**: YOLO 画像分類。
- **ORT_Inspect**: ImageJ 上で ONNX モデルのメタデータを確認するユーティリティ。
- **OrtUtil**: 前処理・NMS・UI ヘルパーを担う共通ユーティリティクラス。
- **MyOrtSession**: YOLO 固有のロジックを内包した OrtSession ラッパー。

### 変更・改善
- 全プラグインのダイアログを `DialogListener` パターンにリファクタリング。  
  マクロ互換性とインタラクティブな UI 動作（フィールドの有効・無効切替）が大幅に向上。
- 画像前処理に統計的デバッグログ（入力範囲・平均値）を追加し、推論入力の検証を容易に。

### テスト・開発環境
- `test/` フォルダに全モデルを対象とした自動推論テストマクロを追加。
- `yolo/` フォルダにモデル検査・ラベル抽出用 Python スクリプトを追加。
- CLI ヘッドレス実行の設定方法を `test/README_CLI_Testing.md` にまとめた。
- `copy_to_plugins.bat` の `pause` を除去し、自動ビルド・テストのフローを改善。

---
*注: 本プロジェクトは OpenCV に依存しない、純 Java + ImageJ 構成で ONNX 推論を実現しています。*
