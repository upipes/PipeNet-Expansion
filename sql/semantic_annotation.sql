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

 Date: 03/07/2026 16:50:51
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for semantic_annotation
-- ----------------------------
DROP TABLE IF EXISTS `semantic_annotation`;
CREATE TABLE `semantic_annotation`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `desc_id` bigint NOT NULL COMMENT 'Related semantic_description id',
  `view_name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Semantic view where the annotation is applied',
  `view_text` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Full view text when the annotation is created',
  `annotated_text` varchar(1000) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Selected text marked by the expert',
  `annotation_effect` varchar(24) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'correct / incorrect',
  `annotation_content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT 'Expert annotation note',
  `update_revise` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT 'Expert revise',
  `annotated_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT 'Annotation creation time',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_semantic_annotation_desc`(`desc_id` ASC) USING BTREE,
  INDEX `idx_semantic_annotation_view`(`desc_id` ASC, `view_name` ASC) USING BTREE,
  INDEX `idx_semantic_annotation_effect`(`annotation_effect` ASC) USING BTREE,
  INDEX `idx_semantic_annotation_time`(`annotated_at` ASC) USING BTREE,
  CONSTRAINT `fk_semantic_annotation_desc` FOREIGN KEY (`desc_id`) REFERENCES `semantic_description` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `chk_semantic_annotation_effect` CHECK (`annotation_effect` in (_utf8mb4'correct',_utf8mb4'inaccurate',_utf8mb4'incorrect'))
) ENGINE = InnoDB AUTO_INCREMENT = 124 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
