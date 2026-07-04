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

 Date: 03/07/2026 16:50:37
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for model_training_run
-- ----------------------------
DROP TABLE IF EXISTS `model_training_run`;
CREATE TABLE `model_training_run`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `source_domain_id` bigint NOT NULL,
  `target_domain_id` bigint NOT NULL,
  `method_name` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT 'Ours',
  `model_name` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `source_dataset` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `target_dataset` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `backbone` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `semantic_generator` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL,
  `embedding_model` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL,
  `optimizer` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `knowledge_items` int UNSIGNED NULL DEFAULT NULL,
  `refinement_iterations` int UNSIGNED NULL DEFAULT NULL,
  `learning_rate` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `batch_size` int UNSIGNED NOT NULL,
  `epochs` int UNSIGNED NOT NULL,
  `accuracy` decimal(5, 2) NOT NULL,
  `class_accuracy` json NOT NULL,
  `method_checkpoint_path` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL,
  `model_description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL,
  `created_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  `updated_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_model_training_run_source_domain`(`source_domain_id` ASC) USING BTREE,
  INDEX `idx_model_training_run_target_domain`(`target_domain_id` ASC) USING BTREE,
  INDEX `idx_model_training_run_method`(`method_name` ASC) USING BTREE,
  INDEX `idx_model_training_run_model`(`model_name` ASC) USING BTREE,
  INDEX `idx_model_training_run_source`(`source_dataset` ASC) USING BTREE,
  INDEX `idx_model_training_run_target`(`target_dataset` ASC) USING BTREE,
  INDEX `idx_model_training_run_backbone`(`backbone` ASC) USING BTREE,
  INDEX `idx_model_training_run_checkpoint`(`method_checkpoint_path` ASC) USING BTREE,
  INDEX `idx_model_training_run_accuracy`(`accuracy` ASC) USING BTREE,
  CONSTRAINT `fk_model_training_run_source_domain` FOREIGN KEY (`source_domain_id`) REFERENCES `area_domain` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_model_training_run_target_domain` FOREIGN KEY (`target_domain_id`) REFERENCES `area_domain` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE = InnoDB AUTO_INCREMENT = 69 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci ROW_FORMAT = DYNAMIC;

SET FOREIGN_KEY_CHECKS = 1;
