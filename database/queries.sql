CREATE DATABASE IF NOT EXISTS breast_cancer_db;
USE breast_cancer_db;

CREATE TABLE IF NOT EXISTS breast_cancer_raw (
    age                    INT,
    race                   VARCHAR(50),
    marital_status         VARCHAR(50),
    t_stage                VARCHAR(10),
    n_stage                VARCHAR(10),
    sixth_stage            VARCHAR(10),
    differentiate          VARCHAR(50),
    grade                  VARCHAR(20),
    a_stage                VARCHAR(20),
    tumor_size             INT,
    estrogen_status        VARCHAR(20),
    progesterone_status    VARCHAR(20),
    regional_node_examined INT,
    regional_node_positive INT,
    survival_months        INT,
    status                 VARCHAR(20)
) ENGINE=ColumnStore;

CREATE TABLE IF NOT EXISTS breast_cancer_processed (
    age                    INT,
    race                   INT,
    marital_status         INT,
    t_stage                INT,
    n_stage                INT,
    sixth_stage            INT,
    differentiate          INT,
    grade                  INT,
    a_stage                INT,
    tumor_size             FLOAT,
    estrogen_status        INT,
    progesterone_status    INT,
    regional_node_examined FLOAT,
    regional_node_positive FLOAT,
    status                 INT
) ENGINE=ColumnStore;

CREATE USER IF NOT EXISTS 'bc_user'@'localhost' IDENTIFIED BY 'bc_password123';
GRANT ALL PRIVILEGES ON breast_cancer_db.* TO 'bc_user'@'localhost';
FLUSH PRIVILEGES;