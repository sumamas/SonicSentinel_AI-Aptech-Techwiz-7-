-- SonicSentinel AI schema (MySQL / MariaDB, XAMPP).
-- You normally do NOT need this file: the app and scripts/init_db.py create
-- every table and add missing columns automatically.

CREATE DATABASE IF NOT EXISTS sonicsentinel CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE sonicsentinel;

CREATE TABLE IF NOT EXISTS users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  full_name VARCHAR(120) NOT NULL,
  email VARCHAR(190) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  role VARCHAR(40) NOT NULL DEFAULT 'normal_user',
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_users_email (email)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS audio_events (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NULL,
  original_filename VARCHAR(255) NOT NULL,
  stored_filename VARCHAR(255) NOT NULL,
  sample_rate INT NOT NULL,
  duration_seconds FLOAT NOT NULL,
  quality_label VARCHAR(30) NOT NULL,
  silence_ratio FLOAT NOT NULL DEFAULT 0,
  clipping_ratio FLOAT NOT NULL DEFAULT 0,
  rms FLOAT NOT NULL DEFAULT 0,
  predicted_class VARCHAR(100) NULL,          -- final decision class
  confidence FLOAT NULL,                      -- final combined confidence
  severity VARCHAR(30) NULL,
  status VARCHAR(40) NOT NULL DEFAULT 'Inspected',
  python_class VARCHAR(100) NULL,             -- raw Python model output
  python_confidence FLOAT NULL,
  gtm_class VARCHAR(100) NULL,                -- raw Teachable Machine output
  gtm_confidence FLOAT NULL,
  agreement_status VARCHAR(40) NULL,
  confidence_difference FLOAT NULL,
  source VARCHAR(20) NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_audio_user (user_id),
  INDEX idx_audio_created (created_at),
  CONSTRAINT fk_audio_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS event_analyses (
  event_id INT PRIMARY KEY,
  data LONGTEXT NOT NULL,                     -- JSON snapshot: metadata, both models, comparison, final
  audio_sha256 VARCHAR(64) NULL,
  INDEX ix_event_analyses_audio_sha256 (audio_sha256),
  CONSTRAINT fk_analysis_event FOREIGN KEY (event_id) REFERENCES audio_events(id) ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS event_actions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  event_id INT NOT NULL,
  actor_id INT NOT NULL,
  action VARCHAR(30) NOT NULL,                -- confirm, correct, acknowledge, dismiss, escalate
  corrected_class VARCHAR(100) NULL,
  comment TEXT NOT NULL,
  recommended_action VARCHAR(1000) NOT NULL DEFAULT '',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX ix_event_actions_event_id (event_id),
  CONSTRAINT fk_action_event FOREIGN KEY (event_id) REFERENCES audio_events(id) ON DELETE CASCADE,
  CONSTRAINT fk_action_user FOREIGN KEY (actor_id) REFERENCES users(id)
) ENGINE=InnoDB;
