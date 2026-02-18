"""
Hospital Database Setup Script
Creates MySQL database with normalized tables and imports CSV data.

Tables:
  - time_periods          : Date, Year, Month, Month_Sin, Month_Cos
  - environmental_factors : Rainfall, Avg_Temperature, Humidity (linked to time_periods)
  - social_indicators     : Festive_Season, Public_Holidays, Public_Awareness_Level (linked to time_periods)
  - disease_cases         : Dengue, Road_Accidents, Heart_Patients, Hadisi_Anthuru,
                            Tuberculosis, Cold, Fever (linked to time_periods)
"""

import csv
import sys
import os

try:
    import pymysql
except ImportError:
    print("[ERROR] pymysql not installed.")
    print("  Run:  pip install pymysql")
    sys.exit(1)

# ─── Connection Config ─────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     "127.0.0.1",
    "user":     "root",
    "password": "",          # XAMPP default: no password
}
DATABASE   = "hospital_db"
CSV_FILE   = "hospital_multi_disease_final.csv"

# ─── SQL Statements ────────────────────────────────────────────────────────────

CREATE_DB = f"CREATE DATABASE IF NOT EXISTS `{DATABASE}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

CREATE_TIME_PERIODS = """
CREATE TABLE IF NOT EXISTS `time_periods` (
    `id`        INT           NOT NULL AUTO_INCREMENT,
    `date`      DATE          NOT NULL UNIQUE,
    `year`      SMALLINT      NOT NULL,
    `month`     TINYINT       NOT NULL,
    `month_sin` DOUBLE        NOT NULL,
    `month_cos` DOUBLE        NOT NULL,
    PRIMARY KEY (`id`)
) ENGINE=InnoDB;
"""

CREATE_ENVIRONMENTAL = """
CREATE TABLE IF NOT EXISTS `environmental_factors` (
    `id`              INT     NOT NULL AUTO_INCREMENT,
    `period_id`       INT     NOT NULL,
    `rainfall`        DOUBLE  NOT NULL COMMENT 'mm',
    `avg_temperature` DOUBLE  NOT NULL COMMENT 'Celsius',
    `humidity`        DOUBLE  NOT NULL COMMENT 'percent',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uq_period` (`period_id`),
    CONSTRAINT `fk_env_period`
        FOREIGN KEY (`period_id`) REFERENCES `time_periods`(`id`)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;
"""

CREATE_SOCIAL = """
CREATE TABLE IF NOT EXISTS `social_indicators` (
    `id`                    INT           NOT NULL AUTO_INCREMENT,
    `period_id`             INT           NOT NULL,
    `festive_season`        TINYINT(1)    NOT NULL COMMENT '0=No, 1=Yes',
    `public_holidays`       TINYINT(1)    NOT NULL COMMENT '0=No, 1=Yes',
    `public_awareness_level` DECIMAL(5,4) NOT NULL COMMENT '0.0 to 1.0',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uq_period` (`period_id`),
    CONSTRAINT `fk_soc_period`
        FOREIGN KEY (`period_id`) REFERENCES `time_periods`(`id`)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;
"""

CREATE_DISEASE = """
CREATE TABLE IF NOT EXISTS `disease_cases` (
    `id`             INT  NOT NULL AUTO_INCREMENT,
    `period_id`      INT  NOT NULL,
    `dengue`         INT  NOT NULL,
    `road_accidents` INT  NOT NULL,
    `heart_patients` INT  NOT NULL,
    `hadisi_anthuru` INT  NOT NULL COMMENT 'Gastroenteritis/Foodborne illness',
    `tuberculosis`   INT  NOT NULL,
    `cold`           INT  NOT NULL,
    `fever`          INT  NOT NULL,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uq_period` (`period_id`),
    CONSTRAINT `fk_dis_period`
        FOREIGN KEY (`period_id`) REFERENCES `time_periods`(`id`)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;
"""

# ─── Insert Queries ────────────────────────────────────────────────────────────

INS_TIME = """
    INSERT IGNORE INTO `time_periods` (date, year, month, month_sin, month_cos)
    VALUES (%s, %s, %s, %s, %s)
"""

INS_ENV = """
    INSERT IGNORE INTO `environmental_factors` (period_id, rainfall, avg_temperature, humidity)
    VALUES (%s, %s, %s, %s)
"""

INS_SOC = """
    INSERT IGNORE INTO `social_indicators`
        (period_id, festive_season, public_holidays, public_awareness_level)
    VALUES (%s, %s, %s, %s)
"""

INS_DIS = """
    INSERT IGNORE INTO `disease_cases`
        (period_id, dengue, road_accidents, heart_patients,
         hadisi_anthuru, tuberculosis, cold, fever)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Hospital DB Setup")
    print("=" * 55)

    # Connect without DB first to create it
    try:
        conn = pymysql.connect(**DB_CONFIG)
    except pymysql.Error as e:
        print(f"[ERROR] Cannot connect to MySQL: {e}")
        sys.exit(1)

    cursor = conn.cursor()

    # --- Force-delete the physical MariaDB data directory to clear orphaned .ibd files ---
    import shutil
    mariadb_db_dir = r"C:\xampp\mysql\data\hospital_db"
    if os.path.exists(mariadb_db_dir):
        shutil.rmtree(mariadb_db_dir, ignore_errors=True)
        print(f"[OK]  Deleted orphaned data directory: {mariadb_db_dir}")

    # Now safely drop and recreate
    cursor.execute(f"DROP DATABASE IF EXISTS `{DATABASE}`;")
    cursor.execute(CREATE_DB)
    cursor.execute(f"USE `{DATABASE}`;")
    print(f"[OK]  Database '{DATABASE}' recreated (tablespace cleared).")

    # Create tables
    for ddl, name in [
        (CREATE_TIME_PERIODS,  "time_periods"),
        (CREATE_ENVIRONMENTAL, "environmental_factors"),
        (CREATE_SOCIAL,        "social_indicators"),
        (CREATE_DISEASE,       "disease_cases"),
    ]:
        cursor.execute(ddl)
        print(f"[OK]  Table '{name}' ready.")

    conn.commit()

    # ── Import CSV ─────────────────────────────────────────────────────────────
    inserted = 0
    skipped  = 0

    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row["Date"][:10]   # keep YYYY-MM-DD only

            # 1) time_periods
            cursor.execute(INS_TIME, (
                date,
                int(row["Year"]),
                int(row["Month"]),
                float(row["Month_Sin"]),
                float(row["Month_Cos"]),
            ))

            # Fetch the ID (existing or just-inserted)
            cursor.execute("SELECT id FROM time_periods WHERE date = %s", (date,))
            period_id = cursor.fetchone()[0]

            # 2) environmental_factors
            cursor.execute(INS_ENV, (
                period_id,
                float(row["Rainfall"]),
                float(row["Avg_Temperature"]),
                float(row["Humidity"]),
            ))

            # 3) social_indicators
            cursor.execute(INS_SOC, (
                period_id,
                int(row["Festive_Season"]),
                int(row["Public_Holidays"]),
                float(row["Public_Awareness_Level"]),
            ))

            # 4) disease_cases
            cursor.execute(INS_DIS, (
                period_id,
                int(row["Dengue"]),
                int(row["Road_Accidents"]),
                int(row["Heart_Patients"]),
                int(row["Hadisi_Anthuru"]),
                int(row["Tuberculosis"]),
                int(row["Cold"]),
                int(row["Fever"]),
            ))

            inserted += 1

    conn.commit()
    cursor.close()
    conn.close()

    print("-" * 55)
    print(f"[DONE] Imported {inserted} records  |  Skipped {skipped} duplicates")
    print("=" * 55)
    print("\nDatabase Schema Summary:")
    print("  hospital_db")
    print("  ├── time_periods          (id, date, year, month, month_sin, month_cos)")
    print("  ├── environmental_factors (id, period_id→, rainfall, avg_temperature, humidity)")
    print("  ├── social_indicators     (id, period_id→, festive_season, public_holidays, public_awareness_level)")
    print("  └── disease_cases         (id, period_id→, dengue, road_accidents, heart_patients,")
    print("                             hadisi_anthuru, tuberculosis, cold, fever)")


if __name__ == "__main__":
    main()
