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

 Date: 03/07/2026 16:50:44
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for original_classifier
-- ----------------------------
DROP TABLE IF EXISTS `original_classifier`;
CREATE TABLE `original_classifier`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `model_name` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'ResNet-50 / ResNet-101 / ViT-S/16',
  `domain_id` bigint NULL DEFAULT NULL COMMENT 'Related training area_domain id',
  `domain_name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Full name of the domain where the classifier is trained',
  `training_type` varchar(24) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'fine_tuned / pre_trained',
  `accuracy` decimal(5, 2) NOT NULL COMMENT 'Training or validation accuracy on the original domain',
  `model_description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT 'Model introduction and training notes',
  `model_file_path` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT 'Saved torch model file path',
  `created_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  `updated_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_original_classifier_model`(`model_name` ASC) USING BTREE,
  INDEX `idx_original_classifier_domain`(`domain_id` ASC) USING BTREE,
  INDEX `idx_original_classifier_domain_name`(`domain_name` ASC) USING BTREE,
  INDEX `idx_original_classifier_type`(`training_type` ASC) USING BTREE,
  INDEX `idx_original_classifier_accuracy`(`accuracy` ASC) USING BTREE,
  CONSTRAINT `fk_original_classifier_domain` FOREIGN KEY (`domain_id`) REFERENCES `area_domain` (`id`) ON DELETE SET NULL ON UPDATE RESTRICT,
  CONSTRAINT `chk_original_classifier_accuracy` CHECK ((`accuracy` >= 0) and (`accuracy` <= 100)),
  CONSTRAINT `chk_original_classifier_model` CHECK (`model_name` in (_utf8mb4'ResNet-50',_utf8mb4'ResNet-101',_utf8mb4'ViT-S/16')),
  CONSTRAINT `chk_original_classifier_type` CHECK (`training_type` in (_utf8mb4'fine_tuned',_utf8mb4'pre_trained'))
) ENGINE = InnoDB AUTO_INCREMENT = 36 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
