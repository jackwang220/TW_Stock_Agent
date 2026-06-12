"""全域設定：從 .env + configs/default.yaml 合併載入。"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

ROOT = Path(__file__).resolve().parents[2]  # TW_Stock_Agent/

load_dotenv(ROOT / ".env")

# curl_cffi（yfinance 1.4+）無法處理含非 ASCII 字元的憑證路徑。
# 用 .env 中的 CURL_CA_BUNDLE，或 fallback 到 certifi 預設路徑的 ASCII 副本。
def _fix_curl_cert() -> None:
    for var in ("CURL_CA_BUNDLE", "SSL_CERT_FILE"):
        if var in os.environ:
            return  # .env 已設定，不需要處理
    try:
        import certifi
        src = Path(certifi.where())
        dst = Path.home() / "cacert.pem"
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            import shutil
            shutil.copy2(src, dst)
        for var in ("CURL_CA_BUNDLE", "SSL_CERT_FILE"):
            os.environ[var] = str(dst)
    except Exception:
        pass

_fix_curl_cert()


def _load_yaml() -> dict:
    p = ROOT / "configs" / "default.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


class Settings(BaseSettings):
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # Shioaji (選用)
    sinopac_apikey: str = Field(default="", alias="SINOPAC_APIKEY")
    sinopac_secretkey: str = Field(default="", alias="SINOPAC_SECRETKEY")
    sinopac_ca_path: str = Field(default="./Sinopac.pfx", alias="SINOPAC_CA_PATH")
    sinopac_ca_password: str = Field(default="", alias="SINOPAC_CA_PASSWORD")
    sinopac_person_id: str = Field(default="", alias="SINOPAC_PERSON_ID")

    finmind_token: str = Field(default="", alias="FINMIND_TOKEN")
    finmind_token2: str = Field(default="", alias="FINMIND_TOKEN2")  # 第二帳號,雙token分攤額度

    discord_webhook_url: str = Field(default="", alias="DISCORD_WEBHOOK_URL")

    class Config:
        populate_by_name = True
        env_file = str(ROOT / ".env")
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_yaml_cfg() -> dict:
    return _load_yaml()


def cfg(key: str, default=None):
    """點號路徑存取 yaml config，例如 cfg('screener.min_volume_ratio')。"""
    parts = key.split(".")
    node = get_yaml_cfg()
    for p in parts:
        if not isinstance(node, dict):
            return default
        node = node.get(p, default)
    return node


# 常用路徑
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
SIGNAL_LOG = DATA_DIR / "signal_log.csv"
CHECKPOINT_DB = DATA_DIR / "scan_checkpoints.db"
COMPANIES_JSON = DATA_DIR / "companies.json"
TW_STOCK_INDEX = DATA_DIR / "tw_stock_index.json"
