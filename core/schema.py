"""元数据库表结构（20 张表）。sqlite3，零额外依赖。

表命名与设计对应《07_数据库与接口设计.md》第 3 节。
"""

from __future__ import annotations

SCHEMA_VERSION = 1

DDL: list[str] = [
    # 1. 用户
    """
    CREATE TABLE IF NOT EXISTS user (
        id TEXT PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        display_name TEXT,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'researcher',
        email TEXT,
        is_active INTEGER NOT NULL DEFAULT 1,
        failed_attempts INTEGER NOT NULL DEFAULT 0,
        locked_until TEXT,
        last_login_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    # 2. 项目
    """
    CREATE TABLE IF NOT EXISTS project (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        scene TEXT DEFAULT 'campus_road',
        app_target TEXT DEFAULT 'sweeper',
        owner_id TEXT,
        status TEXT NOT NULL DEFAULT 'active',
        config TEXT DEFAULT '{}',
        is_deleted INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    # 3. 项目成员
    """
    CREATE TABLE IF NOT EXISTS project_member (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        role_in_project TEXT NOT NULL DEFAULT 'developer',
        joined_at TEXT NOT NULL,
        UNIQUE (project_id, user_id)
    )
    """,
    # 4. 数据集
    """
    CREATE TABLE IF NOT EXISTS dataset (
        id TEXT PRIMARY KEY,
        project_id TEXT,
        name TEXT NOT NULL,
        task_type TEXT NOT NULL,
        scene TEXT,
        class_names TEXT DEFAULT '[]',
        foreground_values TEXT DEFAULT '[]',
        status TEXT NOT NULL DEFAULT 'draft',
        latest_version TEXT,
        description TEXT,
        created_by TEXT,
        is_deleted INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    # 5. 数据集版本
    """
    CREATE TABLE IF NOT EXISTS dataset_version (
        id TEXT PRIMARY KEY,
        dataset_id TEXT NOT NULL,
        version TEXT NOT NULL,
        content_hash TEXT,
        image_count INTEGER NOT NULL DEFAULT 0,
        annotation_count INTEGER NOT NULL DEFAULT 0,
        stats TEXT DEFAULT '{}',
        split_ratio TEXT,
        split_seed INTEGER,
        split_manifest_uri TEXT,
        validation_report TEXT DEFAULT '{}',
        source_root TEXT,
        frozen_at TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (dataset_id, version)
    )
    """,
    # 6. 图像
    """
    CREATE TABLE IF NOT EXISTS image (
        id TEXT PRIMARY KEY,
        dataset_version_id TEXT NOT NULL,
        uri TEXT NOT NULL,
        mask_uri TEXT,
        sha256 TEXT,
        width INTEGER,
        height INTEGER,
        split TEXT DEFAULT 'train',
        source TEXT DEFAULT 'import',
        capture_session_id TEXT,
        illum TEXT,
        target_size_group TEXT,
        meta TEXT DEFAULT '{}',
        created_at TEXT NOT NULL
    )
    """,
    # 7. 深度图
    """
    CREATE TABLE IF NOT EXISTS depth_map (
        id TEXT PRIMARY KEY,
        image_id TEXT NOT NULL UNIQUE,
        uri TEXT NOT NULL,
        encoding TEXT NOT NULL DEFAULT 'png16',
        depth_scale REAL NOT NULL DEFAULT 0.001,
        valid_ratio REAL,
        min_m REAL,
        max_m REAL,
        mean_m REAL,
        calibration_id TEXT,
        created_at TEXT NOT NULL
    )
    """,
    # 8. 标注
    """
    CREATE TABLE IF NOT EXISTS annotation (
        id TEXT PRIMARY KEY,
        image_id TEXT NOT NULL,
        task_type TEXT NOT NULL,
        source_format TEXT,
        payload TEXT DEFAULT '{}',
        is_verified INTEGER NOT NULL DEFAULT 0,
        annotator_id TEXT,
        quality_flags TEXT DEFAULT '[]',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    # 9. 模型
    """
    CREATE TABLE IF NOT EXISTS model (
        id TEXT PRIMARY KEY,
        project_id TEXT,
        name TEXT NOT NULL,
        task_type TEXT NOT NULL,
        framework TEXT,
        arch TEXT,
        variant TEXT,
        input_size TEXT,
        num_classes INTEGER,
        class_names TEXT DEFAULT '[]',
        weights_uri TEXT,
        weights_sha256 TEXT,
        source_filename TEXT,
        weight_origin TEXT DEFAULT 'imported',
        train_experiment_id TEXT,
        metrics TEXT DEFAULT '{}',
        resource TEXT DEFAULT '{}',
        size_bytes INTEGER DEFAULT 0,
        notes TEXT,
        is_deleted INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    # 10. 相机设备
    """
    CREATE TABLE IF NOT EXISTS camera_device (
        id TEXT PRIMARY KEY,
        model TEXT,
        serial TEXT UNIQUE,
        firmware TEXT,
        usb_type TEXT,
        default_resolution TEXT,
        default_fps INTEGER,
        filters TEXT DEFAULT '[]',
        last_seen_at TEXT,
        created_at TEXT NOT NULL
    )
    """,
    # 11. 标定
    """
    CREATE TABLE IF NOT EXISTS calibration (
        id TEXT PRIMARY KEY,
        camera_id TEXT,
        name TEXT NOT NULL,
        intrinsics TEXT DEFAULT '{}',
        extrinsics TEXT DEFAULT '{}',
        depth_scale REAL,
        rms_reprojection_px REAL,
        depth_check TEXT DEFAULT '{}',
        file_uri TEXT,
        file_sha256 TEXT,
        is_default INTEGER NOT NULL DEFAULT 0,
        created_by TEXT,
        created_at TEXT NOT NULL
    )
    """,
    # 12. 实验
    """
    CREATE TABLE IF NOT EXISTS experiment (
        id TEXT PRIMARY KEY,
        project_id TEXT,
        name TEXT NOT NULL,
        task_type TEXT NOT NULL,
        kind TEXT NOT NULL DEFAULT 'infer',
        status TEXT NOT NULL DEFAULT 'draft',
        dataset_version_id TEXT,
        model_id TEXT,
        calibration_id TEXT,
        camera_id TEXT,
        config_snapshot TEXT DEFAULT '{}',
        run_manifest TEXT DEFAULT '{}',
        metrics TEXT DEFAULT '{}',
        progress REAL DEFAULT 0,
        started_at TEXT,
        finished_at TEXT,
        duration_sec REAL,
        artifact_dir TEXT,
        log_uri TEXT,
        error_message TEXT,
        created_by TEXT,
        is_deleted INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    # 13. 检测结果
    """
    CREATE TABLE IF NOT EXISTS detection_result (
        id TEXT PRIMARY KEY,
        experiment_id TEXT NOT NULL,
        image_id TEXT,
        frame_number INTEGER,
        frame_uri TEXT,
        boxes TEXT DEFAULT '[]',
        num_boxes INTEGER DEFAULT 0,
        is_miss INTEGER DEFAULT 0,
        weight_mode TEXT,
        infer_ms REAL,
        preprocess_ms REAL,
        postprocess_ms REAL,
        created_at TEXT NOT NULL
    )
    """,
    # 14. 分割结果
    """
    CREATE TABLE IF NOT EXISTS segmentation_result (
        id TEXT PRIMARY KEY,
        experiment_id TEXT NOT NULL,
        image_id TEXT,
        frame_number INTEGER,
        mask_uri TEXT,
        overlay_uri TEXT,
        threshold REAL,
        open_ks INTEGER,
        close_ks INTEGER,
        mask_area_ratio REAL,
        num_contours INTEGER,
        boundary_length_px INTEGER,
        centroid TEXT,
        is_empty INTEGER DEFAULT 0,
        is_suspicious INTEGER DEFAULT 0,
        miou REAL,
        dice REAL,
        precision_ REAL,
        recall_ REAL,
        f1 REAL,
        infer_ms REAL,
        created_at TEXT NOT NULL
    )
    """,
    # 15. 测距结果（合并项目 A 的 measurements.csv 与项目 B 的 distance_records.csv）
    """
    CREATE TABLE IF NOT EXISTS ranging_result (
        id TEXT PRIMARY KEY,
        experiment_id TEXT,
        detection_result_id TEXT,
        segmentation_result_id TEXT,
        model_id TEXT,
        model_name TEXT,
        task_type TEXT NOT NULL,
        strategy TEXT NOT NULL,
        target_kind TEXT NOT NULL,
        metric_kind TEXT NOT NULL,
        category TEXT,
        size_group TEXT,
        depth_m REAL,
        lateral_m REAL,
        euclidean_m REAL,
        raw_depth_m REAL,
        corrected_depth_m REAL,
        valid_points INTEGER,
        total_points INTEGER,
        sample_points TEXT DEFAULT '[]',
        distance_actual_m REAL,
        actual_source TEXT,
        error_m REAL,
        abs_error_m REAL,
        rel_error REAL,
        illumination TEXT,
        distance_plan_m REAL,
        calibration_applied TEXT,
        frame_number INTEGER,
        record_time TEXT,
        media_uri TEXT,
        notes TEXT,
        source TEXT DEFAULT 'platform',
        created_at TEXT NOT NULL
    )
    """,
    # 16. 误差分析
    """
    CREATE TABLE IF NOT EXISTS error_analysis (
        id TEXT PRIMARY KEY,
        experiment_id TEXT,
        scope TEXT NOT NULL DEFAULT 'project',
        task_type TEXT,
        strategy TEXT,
        group_by TEXT DEFAULT '[]',
        filters TEXT DEFAULT '{}',
        sample_count INTEGER DEFAULT 0,
        valid_count INTEGER DEFAULT 0,
        miss_count INTEGER DEFAULT 0,
        miss_rate REAL,
        mae_m REAL,
        rmse_m REAL,
        mre REAL,
        median_error_m REAL,
        max_error_m REAL,
        std_error_m REAL,
        hit_rate REAL,
        by_distance TEXT DEFAULT '[]',
        by_category TEXT DEFAULT '[]',
        by_model TEXT DEFAULT '[]',
        by_size TEXT DEFAULT '[]',
        by_illumination TEXT DEFAULT '[]',
        outliers TEXT DEFAULT '[]',
        calibration_method TEXT,
        calibration_params TEXT DEFAULT '{}',
        rmse_before REAL,
        rmse_after REAL,
        is_recommended INTEGER DEFAULT 0,
        created_at TEXT NOT NULL
    )
    """,
    # 17. 校正方案
    """
    CREATE TABLE IF NOT EXISTS calibration_plan (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        task_type TEXT,
        strategy TEXT,
        model_id TEXT,
        category TEXT,
        method TEXT NOT NULL,
        params TEXT DEFAULT '{}',
        sample_count INTEGER,
        rmse_before REAL,
        rmse_after REAL,
        is_active INTEGER DEFAULT 0,
        created_at TEXT NOT NULL
    )
    """,
    # 18. 报告
    """
    CREATE TABLE IF NOT EXISTS report (
        id TEXT PRIMARY KEY,
        experiment_id TEXT,
        project_id TEXT,
        name TEXT NOT NULL,
        template TEXT NOT NULL,
        sections TEXT DEFAULT '[]',
        params TEXT DEFAULT '{}',
        format TEXT NOT NULL,
        file_uri TEXT,
        file_size INTEGER DEFAULT 0,
        share_token TEXT,
        status TEXT NOT NULL DEFAULT 'done',
        created_by TEXT,
        created_at TEXT NOT NULL
    )
    """,
    # 19. 审计日志
    """
    CREATE TABLE IF NOT EXISTS audit_log (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        username TEXT,
        action TEXT NOT NULL,
        resource_type TEXT,
        resource_id TEXT,
        result TEXT DEFAULT 'success',
        ip TEXT,
        detail TEXT DEFAULT '{}',
        created_at TEXT NOT NULL
    )
    """,
    # 20. 任务队列
    """
    CREATE TABLE IF NOT EXISTS task_queue (
        id TEXT PRIMARY KEY,
        experiment_id TEXT,
        task_type TEXT NOT NULL,
        payload TEXT DEFAULT '{}',
        priority INTEGER NOT NULL DEFAULT 5,
        status TEXT NOT NULL DEFAULT 'queued',
        worker_id TEXT,
        gpu_index INTEGER,
        enqueued_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        retry_count INTEGER DEFAULT 0,
        error TEXT
    )
    """,
]

INDEXES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_project_owner ON project(owner_id)",
    "CREATE INDEX IF NOT EXISTS idx_member_project ON project_member(project_id)",
    "CREATE INDEX IF NOT EXISTS idx_dataset_project ON dataset(project_id)",
    "CREATE INDEX IF NOT EXISTS idx_dv_dataset ON dataset_version(dataset_id)",
    "CREATE INDEX IF NOT EXISTS idx_image_dv ON image(dataset_version_id, split)",
    "CREATE INDEX IF NOT EXISTS idx_image_session ON image(capture_session_id)",
    "CREATE INDEX IF NOT EXISTS idx_annotation_image ON annotation(image_id, task_type)",
    "CREATE INDEX IF NOT EXISTS idx_model_task ON model(task_type, arch)",
    "CREATE INDEX IF NOT EXISTS idx_model_sha ON model(weights_sha256)",
    "CREATE INDEX IF NOT EXISTS idx_experiment_project ON experiment(project_id, task_type, status)",
    "CREATE INDEX IF NOT EXISTS idx_experiment_created ON experiment(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_det_exp ON detection_result(experiment_id, is_miss)",
    "CREATE INDEX IF NOT EXISTS idx_seg_exp ON segmentation_result(experiment_id)",
    "CREATE INDEX IF NOT EXISTS idx_ranging_exp ON ranging_result(experiment_id, task_type)",
    "CREATE INDEX IF NOT EXISTS idx_ranging_group ON ranging_result(task_type, category, distance_plan_m)",
    "CREATE INDEX IF NOT EXISTS idx_ranging_model ON ranging_result(model_name, distance_plan_m)",
    "CREATE INDEX IF NOT EXISTS idx_ranging_illum ON ranging_result(illumination)",
    "CREATE INDEX IF NOT EXISTS idx_ranging_source ON ranging_result(source)",
    "CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_task_status ON task_queue(status, priority)",
]
