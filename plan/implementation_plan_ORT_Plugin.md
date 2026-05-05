# ORT Plugin for ImageJ: 実装計画書

本書は ImageJ 用 ONNX Runtime 推論プラグインの実装契約を記述する。  
本書に従えば、別の実装者（または生成AI）が同等のコードを再現できることを目指す。

---

## 1. プロジェクト概要

### 1.1 目的

OpenCV DNN ベースのプラグイン（`OCV_NetFromOnnx_*`）の後継として、  
ONNX Runtime（ORT）を推論エンジンに採用した ImageJ プラグイン群を実装する。  
OpenCV への依存を完全に排除し、依存関係を最小化する。

### 1.2 ライセンス

**MIT License**

### 1.3 依存ライブラリ

| ライブラリ | バージョン | 用途 |
|---|---|---|
| `ij.jar` | 既存 | ImageJ 本体 |
| `onnxruntime.jar` | 1.25.1 | ONNX 推論エンジン |

**OpenCV は一切使用しない。** 前処理・後処理はすべて純 Java + ImageJ API で実装する。

### 1.4 ビルドシステム

**Maven**。`pom.xml` で依存関係を管理し、`maven-shade-plugin` を使用して依存ライブラリ（ONNX Runtime）を同梱した Fat JAR を生成する。


---

## 2. ファイル構成

```
[プラグインディレクトリ]
├── OrtUtil.java                  // 共通ユーティリティ（前処理・後処理部品）
├── MyOrtSession.java             // 1モデル分のセッション情報保持
├── ORT_Inspect.java              // モデル構造の解析・表示
├── ORT_1st_ReadAndRelease.java   // モデルの読み込み・解放管理
├── ORT_2nd_Detection.java        // 物体検出推論
├── ORT_2nd_Classify.java         // 分類推論
└── ORT_2nd_Pose.java             // 姿勢推定推論
```

### 2.1 拡張性の方針

将来追加が想定されるファイル（本リリースでは実装しない）：

```
ORT_2nd_Heatmap.java             // 異常検知（EfficientAD 等）
ORT_2nd_Segment.java             // セグメンテーション（SAM 等）
```

> **Note:** E2E 物体検出（`YOLO_Object_E2E`）および E2E ポーズ推定（`YOLO_Pose_E2E`）は v0.1.0 で `ORT_2nd_Detection` / `ORT_2nd_Pose` 内に統合済み。

コード内では `ModelType` を enum で管理し、switch 文で分岐する設計とする。  
新しいタスクは enum 値と対応クラスを追加するだけで拡張できるようにする。

---

## 3. 共通設計規約

### 3.1 全プラグイン共通

- 全クラスは `ExtendedPlugInFilter` を実装する
- コード内コメントは**英語**
- 全 `ORT_2nd_*` に `enable_log`（boolean）チェックボックスを設ける
- `enable_log = false` のとき `IJ.log()` の呼び出しを一切行わない（マクロ自動実行対応）
- ~~全 `ORT_2nd_*` に `enable_refresh_data`（boolean）チェックボックスを設ける~~ → v0.1.0 では未実装

### 3.2 スロット共通仕様

- スロット数: **5（インデックス 0〜4）**
- スロットの選択 UI: **番号のみ** `[0 ▼]`（マクロ互換性のため）
- ダイアログ下部に現在のロード状態をラベルで表示する:
  ```
  0: yolov8n.onnx
  1: yolov8s.onnx
  2: (empty)
  3: (empty)
  4: (empty)
  ```
- ラベルは `gd.addMessage()` で実装する

### 3.3 ResultsTable 共通列

全 `ORT_2nd_*` の ResultsTable には以下の列を必ず含める：

| 列名 | 内容 |
|---|---|
| `Image` | `imp.getTitle()` |
| `Slot` | スロット番号（int） |
| `ModelName` | `MyOrtSession.modelName`（ファイル名のみ、パスなし） |
| `InferenceTime(ms)` | 推論時間（`System.currentTimeMillis()` で計測） |

タスク固有の列はこの後に続ける（各クラスの仕様を参照）。

### 3.4 引用符の自動除去（`trimQuotes`）

ファイルパス入力フィールドの文字列は以下の処理を行う。  
`OrtUtil.java` に `trimQuotes(String)` として実装し、全クラスから参照する。

```
1. trim() で前後の空白を除去
2. 先頭と末尾が " で囲まれていれば除去
3. 先頭と末尾が ' で囲まれていれば除去
```

---

## 4. `OrtUtil.java`

### 4.1 責務

- `OrtEnvironment` のシングルトン保持
- スロット配列（`MyOrtSession[]`）の保持と管理
- 前処理メソッド群
- 共通ユーティリティメソッド

### 4.2 フィールド

```java
public static final int MAX_SLOTS = 5;
private static OrtEnvironment env = null;
private static MyOrtSession[] slots = new MyOrtSession[MAX_SLOTS];
```

### 4.3 主要メソッド

#### `getEnv()`

```java
public static OrtEnvironment getEnv()
```

- `env == null` のとき `OrtEnvironment.getEnvironment()` で初期化
- 以降はシングルトンを返す

#### `getSlot(int index)` / `setSlot(int index, MyOrtSession session)`

スロットの参照・設定。範囲外の場合は `IllegalArgumentException`。

#### `getSlotStatusText()`

```java
public static String getSlotStatusText()
```

ダイアログ下部表示用の文字列を返す。  
例:
```
0: yolov8n.onnx
1: (empty)
2: (empty)
3: (empty)
4: (empty)
```

#### `preprocess(ImageProcessor ip, int targetW, int targetH)`

```java
public static float[] preprocess(ImageProcessor ip, int targetW, int targetH)
```

- letterbox リサイズ（アスペクト比維持、余白は 114,114,114 でパディング）
- `ColorProcessor` の `int[]`（ARGB）から RGB チャンネルを抽出
- `/255.0f` で正規化
- HWC → NCHW に変換
- 戻り値: `float[3 * targetH * targetW]`（NCHW 順）

**letterbox の実装方針:**

```
1. スケール = min(targetW / srcW, targetH / srcH)
2. scaledW = (int)(srcW * scale), scaledH = (int)(srcH * scale)
3. ImageProcessor.resize(scaledW, scaledH) でリサイズ
4. padLeft = (targetW - scaledW) / 2, padTop = (targetH - scaledH) / 2
5. ColorProcessor(targetW, targetH) を生成し Color(114,114,114) で fill
6. copyBits(resized, padLeft, padTop, Blitter.COPY)
```

ARGB → RGB 変換:
```java
int pixel = pixels[i];
int r = (pixel >> 16) & 0xFF;
int g = (pixel >>  8) & 0xFF;
int b =  pixel        & 0xFF;
```

NCHW 変換:
```java
nchw[0 * H * W + i] = r / 255.0f;  // R channel
nchw[1 * H * W + i] = g / 255.0f;  // G channel
nchw[2 * H * W + i] = b / 255.0f;  // B channel
```

#### `trimQuotes(String str)`

3.4 節の仕様を実装。

#### `nms(List<float[]> boxes, float nmsThreshold)`

カスタム NMS 実装。`float[]` は `[x, y, w, h, confidence, classId]` の形式。  
IoU ベースで重複除去。戻り値は残存した `float[]` のリスト。

---

## 5. `MyOrtSession.java`

### 5.1 責務

1モデル分の ORT セッション情報と推論ロジックを保持する。

### 5.2 フィールド

```java
private OrtSession session;
private String modelPath;
private String modelName;        // ファイル名のみ（パスなし）
private ModelType modelType;
private CoordFormat coordFormat;
private int inputWidth;
private int inputHeight;
private String[] classNames;     // null の場合は index 番号で代替
private int numClasses;
private boolean loaded;
```

### 5.3 enum 定義

```java
public enum ModelType {
    YOLO,
    YOLOX,
    CLASSIFICATION,
    POSE,
    // 将来拡張用
    // HEATMAP,
    // SEGMENTATION,
    // YOLO_E2E,
}

public enum CoordFormat {
    YOLO_PIXEL,
    YOLO_NORMALIZED,
    YOLO_OBJECT_E2E,   // E2E 物体検出（YOLOv10 等）: NMS 不要
    YOLOX_UNDECODED,
    YOLO_POSE,
    YOLO_POSE_E2E,     // E2E ポーズ推定: NMS 不要
}
```

### 5.4 主要メソッド

#### `load(String modelPath, ModelType type, CoordFormat coord)`

- `OrtSession` を生成してフィールドに保持
- 入力テンソルの Shape を `session.getInputInfo()` から自動取得
- Shape が動的（`-1`）の場合は `inputWidth = -1`, `inputHeight = -1` をセット
- `loaded = true` にする

#### `isLoaded()`

`loaded` フラグを返す。

#### `release()`

`session.close()` を呼び出し、`loaded = false` にする。

#### `runInference(float[] inputData)`

```java
public OrtSession.Result runInference(float[] inputData) throws OrtException
```

- `OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), new long[]{1, 3, inputHeight, inputWidth})` でテンソル生成
- `session.getInputNames().iterator().next()` で入力名を動的取得
- `session.run(Collections.singletonMap(inputName, tensor))` を実行
- テンソルを `close()` してから Result を返す

#### `getOutputShape()`

```java
public long[] getOutputShape() throws OrtException
```

`session.getOutputInfo()` から最初の出力テンソルの Shape を返す。

#### `resolveClassName(int idx)`

```java
public String resolveClassName(int idx)
```

クラス名を解決して返す。`classNames` が null または範囲外の場合は `"class_N"` にフォールバック。

#### ゲッター群

`getModelName()`, `getModelType()`, `getCoordFormat()`, `getInputWidth()`,  
`getInputHeight()`, `getClassNames()`, `getNumClasses()`, `isLoaded()`

---

## 6. `ORT_Inspect.java`

### 6.1 責務

ONNX モデルファイルの入出力構造を解析し `IJ.log` に出力する。  
スロットとは独立して動作する（ファイル直接指定）。

### 6.2 GUI（`GenericDialog`）

```
[ダイアログタイトル] Inspect ONNX Model

model_path:       [___________________________]   // ファイルフィールド
show_metadata:    □                               // チェックボックス
```

- `model_path`: テキストフィールド（`gd.addFileField()`）。`trimQuotes()` 適用
- `show_metadata`: false がデフォルト

### 6.3 バリデーション

- ファイルが存在しない場合: `IJ.error()` で終了
- ORT の setup: `NO_IMAGE_REQUIRED` を返す

### 6.4 処理（`run()`）

1. `OrtUtil.getEnv()` で `OrtEnvironment` 取得
2. `OrtSession` を生成（Inspect 専用、終了後に必ず `session.close()`）
3. `session.getInputInfo()` で入力テンソル情報を取得
4. `session.getOutputInfo()` で出力テンソル情報を取得
5. `session.getMetadata()` でメタデータ取得（`show_metadata = true` のとき）

### 6.5 出力（`IJ.log`）

```
============================================================
Inspecting ONNX model: yolov8n.onnx
------------------------------------------------------------
[Inputs]
  Name: images
  Shape: [?, 3, 640, 640]
  Type: FLOAT

[Outputs]
  Name: output0
  Shape: [?, 84, 8400]
  Type: FLOAT

[OpSet]
  Version: 17
------------------------------------------------------------
[Metadata]                        ← show_metadata = true のときのみ
  producer_name: pytorch
  domain: ...
  custom: { names: "cat\ndog\n..." }
============================================================
```

**Shape 表示のルール:**
- `-1` は `?` に変換して表示する

**OpSet の取得方法:**
- `session.getMetadata().getCustomMetadataMap()` に含まれない場合は "N/A" と表示

---

## 7. `ORT_1st_ReadAndRelease.java`

### 7.1 責務

モデルの読み込み・解放を一元管理する。  
コンボボックスで操作タイプを切り替える。

### 7.2 GUI（`GenericDialog`）

```
[ダイアログタイトル] Model Management

action:       [ read_model ▼ ]    // コンボボックス
              [ release_model   ]
              [ release_all     ]

model_path:   [___________________________]   // addFileField
model_format: [ YOLO_Object_Pixel ▼ ]        // addChoice
input_width:  640                             // addNumericField
input_height: 640                             // addNumericField
slot:         [ 0 ▼ ]                         // addChoice (0〜4)
enable_log:   ☑                               // addCheckbox

──────────────────────────────
（addMessage で現在のロード状態を表示）
0: yolov8n.onnx
1: (empty)
2: (empty)
3: (empty)
4: (empty)
```

**モデルフォーマット選択肢（`FORMAT_LABELS`）:**

```java
private static final String[] FORMAT_LABELS = {
    "YOLO_Object_Pixel",
    "YOLO_Object_Normalized",
    "YOLO_Object_E2E",       // v0.1.0 追加
    "YOLO_Class",
    "YOLO_Pose",
    "YOLO_Pose_E2E",
    "YOLOX_Object_Undecoded",
};
```

### 7.3 DialogListener による動的グレーアウト

`DialogListener` を実装し、`action` の選択に応じてフィールドを切り替える：

| action | model_path | model_format | input_width/height | slot |
|---|---|---|---|---|
| read_model | 有効 | 有効 | 有効 | 有効 |
| release_model | 無効 | 無効 | 無効 | 有効 |
| release_all | 無効 | 無効 | 無効 | 無効 |

**実装上の注意:** `DialogListener` が正常動作しない場合はグレーアウトを無視し、  
`run()` 内でバリデーションのみ行う（動作を止めない方針）。

### 7.4 入力サイズの自動取得

`read_model` 選択時:

1. モデルをロードして `session.getInputInfo()` で Shape を取得
2. Shape が固定値（`> 0`）なら `input_width` / `input_height` フィールドに反映（ダイアログでは表示のみ、自動入力はしない）
3. Shape が動的（`-1`）なら:
   - `IJ.showStatus("Dynamic shape detected. Please enter input size.")` を表示
   - ダイアログの `input_width` / `input_height` フィールドでユーザーが手動入力

### 7.5 バリデーション

| 条件 | 処理 |
|---|---|
| `read_model` かつファイルが存在しない | `IJ.error()` |
| `read_model` かつ input_width/height <= 0 | `IJ.error()` |
| `read_model` かつ指定スロットに既にモデルあり | `IJ.showMessage()` で警告後、上書き（ユーザー確認） |
| `release_model` かつ指定スロットが空 | `IJ.error()` |
| `release_all` | 確認なしで全スロット解放 |

### 7.6 ログ出力（`enable_log = true`）

**read_model 成功時:**
```
============================================================
Model Load Complete:
  Slot: 0
  File: yolov8n.onnx
  Format: YOLO_Object_Pixel
  Input Size: 640 x 640

[Inputs]
  Name: images
  Shape: [?, 3, 640, 640]
  Type: FLOAT

[Outputs]
  Name: output0
  Shape: [?, 84, 8400]
  Type: FLOAT
============================================================
```

**release_model 成功時:**
```
Slot 0: released. (yolov8n.onnx)
```

**release_all 成功時:**
```
All slots released.
```

---

## 8. `ORT_2nd_Detection.java`

### 8.1 対応フォーマット

```
YOLO_Object_Pixel
YOLO_Object_Normalized
YOLOX_Object_Undecoded
```

上記以外のフォーマットがロードされているスロットを選択した場合: `IJ.error()` で終了。

### 8.2 GUI（`GenericDialog`）

```
[ダイアログタイトル] Detection Inference

slot:               [ 0 ▼ ]
score_threshold:    0.25
nms_threshold:      0.45
show_results_table: ☑
show_roi_manager:   ☑
enable_log:         ☑

──────────────────────────────
（addMessage でロード状態表示）
```

### 8.3 NMS の扱い

- `CoordFormat.YOLOX_UNDECODED` または将来の E2E フォーマット → NMS スキップ
- それ以外 → `OrtUtil.nms()` で独自 NMS を適用

### 8.4 出力テンソルの解釈

**YOLO_Object_Pixel / YOLO_Object_Normalized:**
- 出力 Shape: `[1, 4+numClasses, numAnchors]` or `[1, numAnchors, 4+numClasses]`
- `resolveOutputShape()` で自動判定・転置

**YOLOX_Object_Undecoded:**
- 出力 Shape: `[1, numAnchors, 5+numClasses]`（objectness あり）

### 8.5 ResultsTable 列

```
Image | Slot | ModelName | Label | Confidence | X | Y | Width | Height | InferenceTime(ms)
```

### 8.6 RoiManager

- BBox を `Roi(x, y, w, h)` で追加
- 名前: `"ラベル名: 0.85"` の形式
- 色: クラス ID から HSB で自動生成（既存踏襲）

---

## 9. `ORT_2nd_Classify.java`

### 9.1 対応フォーマット

```
YOLO_Class
```

上記以外のフォーマットのスロットを選択した場合: `IJ.error()` で終了。

### 9.2 GUI（`GenericDialog`）

```
[ダイアログタイトル] Classification Inference

slot:               [ 0 ▼ ]
top_k:              5
show_results_table: ☑
enable_log:         ☑

──────────────────────────────
（addMessage でロード状態表示）
```

### 9.3 ROI 対応

- `imp.getRoi()` が `RECTANGLE` タイプなら `ip.crop()` で切り出して推論
- ROI なし（または RECTANGLE 以外）なら画像全体を推論

### 9.4 クラス名の解決（優先順位）

1. `MyOrtSession.classNames` に値があれば使用（外部ファイルから `ORT_1st_ReadAndRelease` でロード済み）
2. なければモデルのメタデータ（`session.getMetadata().getCustomMetadataMap()`）から `"names"` キーを探す
3. それもなければ `"class_0"`, `"class_1"`, ... を使用

**外部クラスファイルの読み込み（`ORT_1st_ReadAndRelease` にて）:**
- 形式: 1行1クラス名
- エンコーディング: UTF-8 優先、失敗時 Shift-JIS で再試行
- `#` で始まる行はコメントとして無視

### 9.5 ResultsTable 列

```
Image | Slot | ModelName | ROI_X | ROI_Y | ROI_W | ROI_H | Rank | Label | Confidence | InferenceTime(ms)
```

- ROI なしの場合は `ROI_X=0, ROI_Y=0, ROI_W=imageWidth, ROI_H=imageHeight`
- Top-N 件分の行を追加する

### 9.6 RoiManager

Classification では RoiManager への追加は行わない。

---

## 10. `ORT_2nd_Pose.java`

### 10.1 対応フォーマット

```
YOLO_Pose
YOLO_Pose_E2E
```

上記以外のフォーマットのスロットを選択した場合: `IJ.error()` で終了。

### 10.2 GUI（`GenericDialog`）

```
[ダイアログタイトル] Pose Inference

slot:               [ 0 ▼ ]
score_threshold:    0.25
nms_threshold:      0.45
kpt_threshold:      0.50
show_roi_manager:   ☑
enable_log:         ☑

──────────────────────────────
（addMessage でロード状態表示）
```

### 10.3 キーポイント仕様

- キーポイント数: **17 固定（COCO 準拠）**
- 各キーポイントのデータ: `[x, y, confidence]`（3 要素）
- `YOLO_Pose`: NMS を適用してから後処理
- `YOLO_Pose_E2E`: NMS スキップ

**スケルトン定義（既存踏襲）:**

```java
int[][] SKELETON = {
    {3,1},{1,2},{2,4},{1,0},{0,2},
    {5,6},
    {5,7},{7,9},
    {6,8},{8,10},
    {5,11},{11,12},{12,6},
    {11,13},{13,15},
    {12,14},{14,16}
};
```

### 10.4 RoiManager

| ROI の種類 | 名前 | 色 |
|---|---|---|
| BBox (`Roi`) | `"N-Box"` | クラス色（HSB） |
| キーポイント群 (`PointRoi`) | `"N-Kpt"` | `Color.YELLOW` |
| スケルトン (`ShapeRoi`) | `"N-Skel"` | `Color.MAGENTA` |

N は検出された姿勢の通し番号（1 始まり）。

### 10.5 ResultsTable 列

```
Image | Slot | ModelName | Label | Confidence | X | Y | Width | Height | Kpt_Avg | Kpt_Min | Kpt_Max | InferenceTime(ms)
```

- `Kpt_Avg/Min/Max`: `kpt_threshold` を超えたキーポイントの confidence の統計値
- 有効キーポイントが 0 件の場合は `Double.NaN`

---

## 11. 前処理の詳細仕様

### 11.1 対象クラス

`OrtUtil.preprocess()` として実装し、全 `ORT_2nd_*` から呼び出す。

### 11.2 入力・出力

```
入力: ImageProcessor ip (ColorProcessor を想定)
      int targetW, int targetH (モデルの入力サイズ)
出力: float[] (長さ = 3 * targetH * targetW, NCHW 順)
```

### 11.3 Letterbox アルゴリズム

```
scale = min(targetW / (double)ip.getWidth(), targetH / (double)ip.getHeight())
scaledW = (int)Math.round(ip.getWidth() * scale)
scaledH = (int)Math.round(ip.getHeight() * scale)
padLeft = (targetW - scaledW) / 2
padTop  = (targetH - scaledH) / 2

new ColorProcessor(targetW, targetH)
  → setColor(new Color(114, 114, 114)) → fill()
  → copyBits(resized, padLeft, padTop, Blitter.COPY)
```

### 11.4 ARGB → NCHW 変換

```java
int[] pixels = (int[]) padded.getPixels();
float[] nchw = new float[3 * targetH * targetW];
for (int i = 0; i < targetH * targetW; i++) {
    int px = pixels[i];
    nchw[0 * targetH * targetW + i] = ((px >> 16) & 0xFF) / 255.0f; // R
    nchw[1 * targetH * targetW + i] = ((px >>  8) & 0xFF) / 255.0f; // G
    nchw[2 * targetH * targetW + i] = ( px        & 0xFF) / 255.0f; // B
}
```

### 11.5 BBox 座標のスケール戻し

letterbox 後の BBox 座標を元画像座標に戻す：

```
xOrig = (xPadded - padLeft) / scale
yOrig = (yPadded - padTop)  / scale
wOrig = w / scale
hOrig = h / scale
```

---

## 12. エラーハンドリング方針

| 状況 | 対処 |
|---|---|
| 推論中の `OrtException` | `IJ.error()` でメッセージ表示、`run()` を中断 |
| スロット未ロード | `IJ.error("Model is not loaded in slot N.")` |
| フォーマット不一致 | `IJ.error("Unsupported format: XXX for this plugin.")` |
| 動的シェイプかつサイズ未指定 | `IJ.showStatus()` で案内 → ダイアログで入力 |
| `OrtSession.close()` の失敗 | ログに記録するが例外は握り潰す |

---

## 13. 既知の非対応事項（将来対応）

- Heatmap 出力モデル（EfficientAD 等）
- セグメンテーションモデル（SAM 等）
- E2E 物体検出（YOLOv10 / RT-DETR 等）
- GPU 推論（`onnxruntime_gpu` への切替）
- バッチ推論（バッチサイズ > 1）
- 大規模 CSV でのパフォーマンス最適化

---

## 14. バージョン

初期バージョン: `0.1.0`  
バージョン定数は `OrtUtil.java` に `public static final String VERSION = "0.1.0";` として定義する。

---

## 15. CLI テスト環境

### 15.1 基本コマンド

プロジェクトディレクトリから `java -jar ij.jar` を実行した場合、ImageJ はカレントディレクトリを基準にプラグインを探してしまう。  
**`-Dplugins.dir`** を明示的に指定することが必要。

```powershell
# ヘッドレス実行（バッチモード）
java "-Dplugins.dir=C:\tools\ImageJ" -Xmx2g -jar "C:\tools\ImageJ\ij.jar" -batch "test/test_all_models.ijm"

# GUI 起動してマクロ実行
java "-Dplugins.dir=C:\tools\ImageJ" -Xmx2g -jar "C:\tools\ImageJ\ij.jar" -macro "test/test_all_models.ijm"
```

### 15.2 ビルドとコピーの手順

```powershell
mvn clean package -DskipTests
.\copy_to_plugins.bat
```

### 15.3 注意点

- `copy_to_plugins.bat` には `pause` を入れないこと（自動テストがブロックされる）
- `-batch` モードでは `rt.show("Results")` 等のウィンドウ表示が GUI と異なる動作をするため、プラグイン側で `if (!IJ.isMacro()) rt.show("Results");` と制御する
- `System.out.println()` による `DEBUG:` ログは `-batch` モードでもコンソールに出力される
- 詳細は `test/README_CLI_Testing.md` を参照。
