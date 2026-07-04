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

 Date: 03/07/2026 16:50:58
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for semantic_category
-- ----------------------------
DROP TABLE IF EXISTS `semantic_category`;
CREATE TABLE `semantic_category`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Category name, such as Cavity, Crack, Normal, Pipeline',
  `domain_id` bigint NOT NULL COMMENT 'Related area_domain id',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `uk_semantic_category_domain_name`(`domain_id` ASC, `name` ASC) USING BTREE,
  INDEX `idx_semantic_category_domain`(`domain_id` ASC) USING BTREE,
  CONSTRAINT `fk_semantic_category_domain` FOREIGN KEY (`domain_id`) REFERENCES `area_domain` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 52 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
