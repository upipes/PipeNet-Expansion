/*
 Navicat Premium Data Transfer

 Source Server         : cross_area_gpr
 Source Server Type    : MySQL
 Source Server Version : 80028
 Source Host           : localhost:3306
 Source Schema         : gpr

 Target Server Type    : MySQL
 Target Server Version : 80028
 File Encoding         : 65001

 Date: 03/07/2026 16:51:05
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for semantic_description
-- ----------------------------
DROP TABLE IF EXISTS `semantic_description`;
CREATE TABLE `semantic_description`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `run_id` bigint NOT NULL COMMENT 'Related semantic_generation_run id',
  `domain_id` bigint NOT NULL COMMENT 'Related area_domain id',
  `category_id` bigint NOT NULL COMMENT 'Related semantic_category id',
  `primary_view` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Most important semantic view',
  `primary_brief_description` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Brief description of primary view for main card display',
  `all_view_brief_descriptions` json NOT NULL COMMENT 'All view brief descriptions as key-value JSON',
  `all_view_detailed_descriptions` json NOT NULL COMMENT 'All view detailed descriptions as key-value JSON',
  `llm_confidence` decimal(5, 2) NOT NULL COMMENT 'LLM confidence score',
  `generated_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT 'Time when this semantic description is obtained',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_semantic_description_run`(`run_id` ASC) USING BTREE,
  INDEX `idx_semantic_description_domain_category`(`domain_id` ASC, `category_id` ASC) USING BTREE,
  INDEX `idx_semantic_description_generated_at`(`generated_at` ASC) USING BTREE,
  INDEX `fk_semantic_description_category`(`category_id` ASC) USING BTREE,
  CONSTRAINT `fk_semantic_description_category` FOREIGN KEY (`category_id`) REFERENCES `semantic_category` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `fk_semantic_description_domain` FOREIGN KEY (`domain_id`) REFERENCES `area_domain` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `fk_semantic_description_run` FOREIGN KEY (`run_id`) REFERENCES `semantic_generation_run` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 1051 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
