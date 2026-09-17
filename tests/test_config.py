from underwriting_engine.config import ensure_dirs, load_config


def test_load_config_reads_default_yaml():
    config = load_config("configs/default.yaml")
    assert config["seed"] == 42
    assert config["model"]["target"] == "expected_loss"
    assert config["model"]["group_col"] == "geography_id"
    assert "pricing" in config
    assert "monitoring" in config


def test_ensure_dirs_creates_expected_directories(tmp_path):
    config = {
        "data": {
            "train_path": str(tmp_path / "data" / "processed" / "train.csv"),
            "holdout_path": str(tmp_path / "data" / "processed" / "holdout.csv"),
        },
        "artifacts": {
            "model_dir": str(tmp_path / "models"),
            "report_dir": str(tmp_path / "reports"),
        },
    }

    ensure_dirs(config)

    assert (tmp_path / "data" / "processed").is_dir()
    assert (tmp_path / "models").is_dir()
    assert (tmp_path / "reports").is_dir()
