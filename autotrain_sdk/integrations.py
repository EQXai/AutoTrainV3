from __future__ import annotations
"""External integrations: model uploads and notifications.

This module provides helper functions invoked by ``JobManager`` when a job
finishes (DONE, FAILED or CANCELED).

Configuration is done mainly via **environment variables** for simplicity.
A future version of the UI can expose dedicated fields that set the same env
vars internally (or write a secrets file).

Current supported integrations
------------------------------
1. Hugging Face Hub upload (requires ``huggingface_hub``)
   Env vars:
     * ``AUTO_HF_TOKEN``    – user access token ("Write" permission)
     * ``AUTO_HF_REPO``     – repo id (e.g. "username/my-model")
     * ``AUTO_HF_PRIVATE``  – "1" to create private repo (optional)

2. Amazon S3 upload (requires ``boto3``)
   Env vars:
     * ``AUTO_S3_BUCKET``   – bucket name
     * ``AUTO_S3_PREFIX``   – optional prefix inside the bucket
     * ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` / ``AWS_DEFAULT_REGION``

3. Generic webhook (e.g. Discord) – HTTP POST with JSON payload
   Env vars:
     * ``AUTO_WEBHOOK_URL`` – full URL

4. Simple e-mail via SMTP (optional, only if all vars are present)
   Env vars:
     * ``AUTO_EMAIL_SMTP``  – host:port (e.g. "smtp.gmail.com:587")
     * ``AUTO_EMAIL_USER``  – SMTP username (or sender address)
     * ``AUTO_EMAIL_PASS``  – SMTP password (app password or token)
     * ``AUTO_EMAIL_TO``    – destination address

5. Remote output directory
   Env vars:
     * ``AUTO_REMOTE_BASE`` – base path where completed runs will be stored
     * ``AUTO_REMOTE_DIRECT`` – "1" to write directly to remote path instead of local and move later
"""

from pathlib import Path
import os
import logging
import json
import smtplib
from email.message import EmailMessage
from typing import Tuple, Any, Dict, List
from shutil import move, rmtree, copy2

import importlib
from .paths import OUTPUT_DIR, get_project_root

# Optional deps – imported lazily

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Column headers that can be sent to Google Sheets. Edit here to keep all
# interfaces (Gradio / CLI / integrations) in sync.

GSHEET_HEADER: List[str] = [
    "Timestamp",
    "Dataset",
    "Profile",
    "Status",
    "Images",
    "Steps",
    "lr",
    "batch",
    "epochs",
    "resolution",
    "network_dim",
    "network_alpha",
    "mixed_precision",
    "FID",
    "CLIP",
    "JobID",
    "Uploads",
]


# ---------------------------------------------------------------------------
# Helper upload functions
# ---------------------------------------------------------------------------

def _upload_to_hf(run_dir: Path) -> Tuple[bool, str]:
    # Optional toggle – if AUTO_HF_ENABLE != "1" skip silently
    if os.getenv("AUTO_HF_ENABLE", "0") != "1":
        return False, "HF upload disabled"

    token = os.getenv("AUTO_HF_TOKEN")
    repo_id = os.getenv("AUTO_HF_REPO")
    if not token or not repo_id:
        return False, "HF token/repo not configured"
    try:
        huggingface_hub = importlib.import_module("huggingface_hub")
    except ModuleNotFoundError:
        return False, "huggingface_hub not installed"

    private = os.getenv("AUTO_HF_PRIVATE", "0") == "1"
    api = huggingface_hub.HfApi(token=token)

    try:
        # Create repo if it does not exist
        if repo_id not in [r.repo_id for r in api.list_repos("model")]:
            api.create_repo(repo_id, token=token, private=private, repo_type="model")
    except Exception:
        # repo may already exist or permission denied; ignore and continue
        pass

    try:
        api.upload_folder(
            folder_path=str(run_dir),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=run_dir.name,
        )
        return True, f"Uploaded to HF: {repo_id}/{run_dir.name}"
    except Exception as e:
        return False, f"HF upload error: {e}"


def _upload_to_s3(run_dir: Path) -> Tuple[bool, str]:
    bucket = os.getenv("AUTO_S3_BUCKET")
    if not bucket:
        return False, "S3 bucket not configured"
    try:
        boto3 = importlib.import_module("boto3")
    except ModuleNotFoundError:
        return False, "boto3 not installed"

    prefix = os.getenv("AUTO_S3_PREFIX", "").lstrip("/")
    s3 = boto3.client("s3")
    try:
        for p in run_dir.rglob("*"):
            if p.is_file():
                key = "/".join(filter(None, [prefix, run_dir.name, str(p.relative_to(run_dir))]))
                s3.upload_file(str(p), bucket, key)
        return True, f"Uploaded to S3: s3://{bucket}/{prefix}/{run_dir.name}"
    except Exception as e:
        return False, f"S3 upload error: {e}"


# ---------------------------------------------------------------------------
# Notification helpers
# ---------------------------------------------------------------------------

def _send_webhook(payload: dict) -> Tuple[bool, str]:
    url = os.getenv("AUTO_WEBHOOK_URL")
    if not url:
        return False, "Webhook URL not configured"
    import requests

    try:
        resp = requests.post(url, json=payload, timeout=10)
        ok = resp.status_code < 300
        return ok, f"Webhook status {resp.status_code}"
    except Exception as e:
        return False, f"Webhook error: {e}"


def _send_email(subject: str, body: str) -> Tuple[bool, str]:
    smtp_cfg = os.getenv("AUTO_EMAIL_SMTP")
    user = os.getenv("AUTO_EMAIL_USER")
    pwd = os.getenv("AUTO_EMAIL_PASS")
    to_addr = os.getenv("AUTO_EMAIL_TO")
    if not all([smtp_cfg, user, pwd, to_addr]):
        return False, "SMTP config missing"
    host, _, port = smtp_cfg.partition(":")
    port = int(port or 587)
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr
    msg.set_content(body)

    try:
        with smtplib.SMTP(host, port) as smtp:
            smtp.starttls()
            smtp.login(user, pwd)
            smtp.send_message(msg)
        return True, "Email sent"
    except Exception as e:
        return False, f"Email error: {e}"


# ---------------------------------------------------------------------------
# Remote output helpers
# ---------------------------------------------------------------------------

def _is_subpath(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _transfer_to_remote(run_dir: Path) -> tuple[bool, str]:
    remote_base = os.getenv("AUTO_REMOTE_BASE")
    if not remote_base:
        return False, "Remote base not configured"

    remote_base_p = Path(remote_base)
    try:
        remote_base_p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return False, f"Cannot create remote base: {e}"

    if _is_subpath(run_dir, remote_base_p):
        return False, "Run already in remote path"

    # Destination folder to store the exported model(s)
    dest_dir = remote_base_p / run_dir.name

    # Collect candidate model files (only top-level first, fallback to recursive)
    model_files = list(run_dir.glob("*.safetensors")) + list(run_dir.glob("*.ckpt"))
    if not model_files:
        model_files = list(run_dir.rglob("*.safetensors")) + list(run_dir.rglob("*.ckpt"))

    if not model_files:
        return False, "No model file found to transfer"

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return False, f"Cannot create destination folder: {e}"

    transferred: list[str] = []
    for mdl_path in model_files:
        tgt = dest_dir / mdl_path.name
        try:
            if tgt.exists():
                tgt.unlink()
            copy2(mdl_path, tgt)
            transferred.append(mdl_path.name)
        except Exception as e:
            return False, f"Copy error: {e}"

    return True, f"Copied {len(transferred)} model file(s) to {dest_dir}"


# ---------------------------------------------------------------------------
# Metric computation (simple FID & CLIPScore)
# ---------------------------------------------------------------------------

def _load_image_paths(folder: Path, limit: int = 50):
    exts = (".png", ".jpg", ".jpeg")
    paths = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    return paths[:limit]


def _compute_and_store_metrics(run_dir: Path, dataset_name: str):
    sample_dir = run_dir / "sample"
    if not sample_dir.exists():
        return

    fake_paths = _load_image_paths(sample_dir)
    real_paths = _load_image_paths(Path(get_project_root()) / "input" / dataset_name)
    if not fake_paths or not real_paths:
        return

    fid_value = None
    clip_value = None

    try:
        import torch
        from torchvision import transforms
        from PIL import Image
        from torchmetrics.image.fid import FrechetInceptionDistance

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fid = FrechetInceptionDistance(feature=64).to(device)

        tf = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])

        with torch.no_grad():
            for p in real_paths:
                img = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                fid.update(img, real=True)
            for p in fake_paths:
                img = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                fid.update(img, real=False)
        fid_value = float(fid.compute().item())
    except Exception:
        pass

    # simple CLIPScore using open_clip
    try:
        import open_clip
        import torch
        from PIL import Image
        from torchvision import transforms

        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        texts = [dataset_name] * len(fake_paths)
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        text_tokens = tokenizer(texts).to(device)
        with torch.no_grad():
            text_feats = model.encode_text(text_tokens)
            text_feats /= text_feats.norm(dim=-1, keepdim=True)

            img_feats = []
            for p in fake_paths:
                img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                feat = model.encode_image(img)
                feat /= feat.norm(dim=-1, keepdim=True)
                img_feats.append(feat)
            img_feats = torch.cat(img_feats, 0)
            clip_sims = (img_feats @ text_feats.T).squeeze()
            clip_value = float(clip_sims.mean().item())
    except Exception:
        pass

    metrics: Dict[str, Any] = {}
    if fid_value is not None:
        metrics["fid"] = fid_value
    if clip_value is not None:
        metrics["clip"] = clip_value

    if metrics:
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


# ---------------------------------------------------------------------------
# Google Sheets helper
# ---------------------------------------------------------------------------

def _append_to_gsheets(job, uploads: list[str] | None = None) -> Tuple[bool, str]:  # type: ignore[valid-type]
    """Append a row with *job* information to a Google Sheet.

    Configuration via env vars:

    * ``AUTO_GSHEET_CRED`` – path to a *service-account* JSON key.
    * ``AUTO_GSHEET_ID``   – Spreadsheet ID (the long hash in the URL).
    * ``AUTO_GSHEET_TAB``  – Optional worksheet name (defaults to first sheet).
    """

    cred_path = os.getenv("AUTO_GSHEET_CRED")
    sheet_id = os.getenv("AUTO_GSHEET_ID")

    # Fallback: read stored config if env vars absent
    if not cred_path or not sheet_id:
        try:
            from .gradio_app import _load_integrations_cfg  # lazy import

            cfg = _load_integrations_cfg()
            cred_path = cred_path or cfg.get("AUTO_GSHEET_CRED")
            sheet_id = sheet_id or cfg.get("AUTO_GSHEET_ID")
            tab_env = os.getenv("AUTO_GSHEET_TAB") or cfg.get("AUTO_GSHEET_TAB")
            if tab_env:
                os.environ["AUTO_GSHEET_TAB"] = str(tab_env)
        except Exception:
            pass
    # set env so future calls have it
    if cred_path:
        os.environ["AUTO_GSHEET_CRED"] = str(cred_path)
    if sheet_id:
        os.environ["AUTO_GSHEET_ID"] = str(sheet_id)

    if not cred_path or not sheet_id:
        return False, "Google Sheets not configured"

    # Optional import – keep it lazy so the dependency is only required if used
    try:
        import gspread  # type: ignore
        from google.oauth2.service_account import Credentials  # type: ignore
        from gspread.utils import rowcol_to_a1
    except ModuleNotFoundError:
        return False, "gspread / google-auth not installed"

    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(cred_path, scopes=scopes)
        client = gspread.authorize(creds)

        # Select worksheet
        try:
            sh = client.open_by_key(sheet_id)
        except Exception as exc:
            # Handle quota exceeded (HTTP 429) or other API errors gracefully
            exc_str = str(exc)
            if "429" in exc_str or getattr(exc, "response", None) and getattr(exc.response, "status", None) == 429:
                return False, "Google Sheets quota exceeded (429) – skipping log entry"
            return False, f"Google Sheets error: {exc_str}"
        tab_name = os.getenv("AUTO_GSHEET_TAB")
        if tab_name:
            try:
                worksheet = sh.worksheet(tab_name)
            except Exception as exc:  # WorksheetNotFound or other
                # Try to create the worksheet automatically
                worksheet = sh.add_worksheet(title=tab_name, rows=1000, cols=20)
        else:
            worksheet = sh.sheet1

        import datetime, toml, json as _json

        ts = datetime.datetime.utcnow().isoformat()
        status_val = getattr(job.status, "value", str(job.status)).lower()

        # ---------------- Dataset images -----------------
        SUP_EXT = {".png", ".jpg", ".jpeg"}
        ds_folder = Path(get_project_root()) / "input" / job.dataset
        if not ds_folder.exists():
            # try external sources mapping
            try:
                src_path = Path(get_project_root()) / ".gradio" / "dataset_sources.json"
                if src_path.exists():
                    mapping = _json.loads(src_path.read_text())
                    ds_ext = mapping.get(job.dataset)
                    if ds_ext:
                        ds_folder = Path(ds_ext)
            except Exception:
                pass

        num_images = 0
        if ds_folder.exists():
            for ext in SUP_EXT:
                num_images += len(list(ds_folder.glob(f"*{ext}")))

        # ---------------- Steps -----------------
        steps_val = getattr(job, "total_steps", None)

        # Training hyperparams from TOML
        lr_val = batch_val = epochs_val = ""
        # extra params
        res_val = dim_val = alpha_val = mp_val = ""
        try:
            cfg = toml.load(job.toml_path)
            lr_val = cfg.get("learning_rate") or cfg.get("lr") or ""
            batch_val = cfg.get("train_batch_size") or cfg.get("batch_size") or ""
            epochs_val = cfg.get("max_train_epochs") or cfg.get("epochs") or ""
            res_val = cfg.get("resolution", "")
            dim_val = cfg.get("network_dim", "")
            alpha_val = cfg.get("network_alpha", "")
            mp_val = cfg.get("mixed_precision", "")
        except Exception:
            pass

        # Metrics
        fid_val = clip_val = ""
        metrics_path = Path(job.run_dir) / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
                fid_val = metrics.get("fid", "")
                clip_val = metrics.get("clip", "")
            except Exception:
                pass

        # ---------------- Select columns -----------------

        # Mapping key → value so we can filter dynamically
        data_map = {
            "Timestamp": ts,
            "Dataset": job.dataset,
            "Profile": job.profile,
            "Status": status_val,
            "Images": num_images,
            "Steps": steps_val,
            "lr": lr_val,
            "batch": batch_val,
            "epochs": epochs_val,
            "resolution": res_val,
            "network_dim": dim_val,
            "network_alpha": alpha_val,
            "mixed_precision": mp_val,
            "FID": fid_val,
            "CLIP": clip_val,
            "JobID": job.id,
            "Uploads": ", ".join(uploads or []),
        }

        # Retrieve user-selected keys (comma-separated) from env or stored cfg
        keys_csv = os.getenv("AUTO_GSHEET_KEYS", "")
        if not keys_csv:
            try:
                from .gradio_app import _load_integrations_cfg  # lazy import to avoid circular top-level

                _cfg = _load_integrations_cfg()
                keys_csv = _cfg.get("AUTO_GSHEET_KEYS", "")
            except Exception:
                pass

        if keys_csv:
            os.environ["AUTO_GSHEET_KEYS"] = str(keys_csv)

        selected_keys = [k.strip() for k in keys_csv.split(",") if k.strip()] if keys_csv else list(GSHEET_HEADER)

        # Ensure we always have at least Timestamp to prevent empty header
        if not selected_keys:
            selected_keys = list(GSHEET_HEADER)

        # Build ordered header and row (use user order)
        header = [k for k in selected_keys if k in GSHEET_HEADER]

        # Always include JobID to ensure we can update rows later
        if "JobID" not in header:
            header.append("JobID")

        row = [data_map.get(k, "") for k in header]

        # -------------- Ensure header row --------------
        # If sheet is empty OR first row doesn't contain "JobID" treat as new
        try:
            header_in_sheet = worksheet.row_values(1)
        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or getattr(exc, "response", None) and getattr(exc.response, "status", None) == 429:
                return False, "Google Sheets quota exceeded (429) – skipping log entry"
            return False, f"Google Sheets error: {exc_str}"

        if not header_in_sheet or "JobID" not in header_in_sheet:
            try:
                worksheet.insert_row(header, index=1)
            except Exception as exc:
                exc_str = str(exc)
                if "429" in exc_str or getattr(exc, "response", None) and getattr(exc.response, "status", None) == 429:
                    return False, "Google Sheets quota exceeded (429) – skipping log entry"
                return False, f"Google Sheets error: {exc_str}"
            header_in_sheet = header  # we just wrote it

        # Build mapping col name → index (1-based)
        col_map = {name: idx + 1 for idx, name in enumerate(header_in_sheet)}

        # -------------- Locate existing row by JobID --------------
        existing_row_idx = None
        if "JobID" in col_map:
            try:
                # only search within the JobID column to avoid false matches
                col_idx = col_map["JobID"]
                # get all existing JobID values (skip header)
                jobid_cells = worksheet.col_values(col_idx)[1:]
                for idx, val in enumerate(jobid_cells, start=2):  # data starts at row 2
                    if str(val).strip() == str(job.id):
                        existing_row_idx = idx
                        break
            except Exception:
                pass

        # Fallback: exhaustive search if not found (mitigates eventual consistency)
        if existing_row_idx is None:
            try:
                found_cell = worksheet.find(str(job.id))  # type: ignore[arg-type]
                if found_cell is not None:
                    existing_row_idx = found_cell.row
            except Exception:
                pass

        if existing_row_idx:
            # update full header width (existing sheet header may be longer)
            from gspread.utils import rowcol_to_a1

            end_col = max(len(header_in_sheet), len(row))
            start_a1 = rowcol_to_a1(existing_row_idx, 1)
            end_a1 = rowcol_to_a1(existing_row_idx, end_col)
            # pad row to end_col length
            row_padded = row + [""] * (end_col - len(row))
            worksheet.update(f"{start_a1}:{end_a1}", [row_padded])
        else:
            try:
                worksheet.append_row(row)
            except Exception as exc:
                exc_str = str(exc)
                if "429" in exc_str or getattr(exc, "response", None) and getattr(exc.response, "status", None) == 429:
                    return False, "Google Sheets quota exceeded (429) – skipping log entry"
                return False, f"Google Sheets error: {exc_str}"

        # -------------- Deduplicate accidental duplicates --------------
        try:
            dup_cells = worksheet.findall(str(job.id))  # type: ignore[arg-type]
            rows_with_id = sorted({c.row for c in dup_cells})
            if len(rows_with_id) > 1:
                # keep first occurrence, delete the rest (from bottom to top)
                for r in reversed(rows_with_id[1:]):
                    try:
                        worksheet.delete_rows(r)
                    except Exception:
                        pass
        except Exception:
            pass

        # Calculate steps if not present and we have epochs & batch & images
        if not steps_val:
            try:
                if epochs_val and batch_val and num_images:
                    from math import ceil
                    steps_val = int(int(epochs_val) * ceil(num_images / int(batch_val)))
            except Exception:
                pass

        steps_val = steps_val or ""

        return True, "Google Sheets row appended"
    except Exception as e:  # noqa: BLE001
        import traceback, textwrap
        tb = traceback.format_exc()
        # include exception class for clarity
        err_msg = f"{e.__class__.__name__}: {e}".strip()
        debug_msg = "Google Sheets error: " + err_msg + "\n" + textwrap.indent(tb, "    ")
        return False, debug_msg


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def handle_job_complete(job) -> None:  # type: ignore[valid-type]
    """Called by :pyclass:`JobManager` when a job finishes or is canceled."""

    # Upload only on successful completion
    uploads: list[str] = []
    status_val = getattr(job.status, "value", str(job.status)).lower()
    if status_val == "done":
        run_dir = Path(job.run_dir)

        for fn in (_upload_to_hf, _upload_to_s3):
            ok, msg = fn(run_dir)
            if ok:
                uploads.append(msg)
            elif "not configured" not in msg:
                uploads.append(msg)  # record errors except silent "not configured"

    # Always try to append to Google Sheets (even for failed/canceled runs)
    ok, msg = _append_to_gsheets(job, uploads)
    if not ok and "not configured" not in msg:
        logger.warning(msg)

    # Build notification payload
    payload = {
        "id": job.id,
        "dataset": job.dataset,
        "profile": job.profile,
        "status": status_val,
        "run_dir": str(job.run_dir),
        "uploads": uploads,
    }
    body = json.dumps(payload, indent=2)
    lines = [f"Job {job.id} finished with status {job.status}."] + uploads
    subject = f"AutoTrainV2 job {job.id} – {job.status}"

    ok, msg = _send_webhook(payload)
    if not ok and "not configured" not in msg:
        logger.warning(msg)

    ok, msg = _send_email(subject, body)
    if not ok and "not configured" not in msg:
        logger.warning(msg)

    # ---- compute and store metrics ----
    try:
        _compute_and_store_metrics(Path(job.run_dir), job.dataset)
    except Exception as e:  # noqa: BLE001
        logger.warning("Metric computation failed: %s", e)

    # ---- remote transfer & cleanup ----
    moved, move_msg = _transfer_to_remote(Path(job.run_dir))
    if moved:
        logger.info(move_msg)
        # delete any remaining local files (run_dir moved already)
    elif "not configured" not in move_msg:
        logger.warning(move_msg)

    # Update registry with final metrics and uploads
    try:
        from .run_registry import upsert as _rr_upsert
        extra = {
            "status": status_val,
            "uploads": uploads,
        }
        metrics_path = Path(job.run_dir) / "metrics.json"
        if metrics_path.exists():
            import json as _json
            extra["metrics"] = _json.loads(metrics_path.read_text())
        _rr_upsert(job, extra)
    except Exception:
        pass 