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

 Date: 03/07/2026 16:51:11
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for semantic_generation_run
-- ----------------------------
DROP TABLE IF EXISTS `semantic_generation_run`;
CREATE TABLE `semantic_generation_run`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `domain_id` bigint NOT NULL COMMENT 'The area_domain id where semantic generation is executed',
  `llm_name` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Selected LLM name',
  `use_expert_knowledge` tinyint(1) NOT NULL DEFAULT 0 COMMENT 'Whether expert knowledge is used',
  `use_image_assist` tinyint(1) NOT NULL DEFAULT 0 COMMENT 'Whether image assistance is used',
  `generated_count` int UNSIGNED NOT NULL DEFAULT 0 COMMENT 'Number of generated category descriptions',
  `status` varchar(24) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT 'success' COMMENT 'success / failed / cancelled / running',
  `generated_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT 'Generate button clicked time',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_semantic_run_domain_time`(`domain_id` ASC, `generated_at` ASC) USING BTREE,
  INDEX `idx_semantic_run_config`(`domain_id` ASC, `llm_name` ASC, `use_expert_knowledge` ASC, `use_image_assist` ASC, `status` ASC) USING BTREE,
  CONSTRAINT `fk_semantic_run_domain` FOREIGN KEY (`domain_id`) REFERENCES `area_domain` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 263 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
