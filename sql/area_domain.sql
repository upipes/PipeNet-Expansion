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

 Date: 03/07/2026 16:49:07
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for area_domain
-- ----------------------------
DROP TABLE IF EXISTS `area_domain`;
CREATE TABLE `area_domain`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `code` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `domain_type` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'road / soil',
  `display_order` int UNSIGNED NOT NULL DEFAULT 0,
  `is_active` tinyint(1) NOT NULL DEFAULT 1,
  `condition_text` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '',
  `road_surface` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL,
  `frequency_min` int UNSIGNED NULL DEFAULT NULL,
  `frequency_max` int UNSIGNED NULL DEFAULT NULL,
  `time_window_ns` int UNSIGNED NULL DEFAULT NULL,
  `sand_percent` decimal(5, 2) NULL DEFAULT NULL,
  `silt_percent` decimal(5, 2) NULL DEFAULT NULL,
  `clay_percent` decimal(5, 2) NULL DEFAULT NULL,
  `water_min` decimal(5, 2) NULL DEFAULT NULL,
  `water_max` decimal(5, 2) NULL DEFAULT NULL,
  `permittivity_min` decimal(7, 3) NULL DEFAULT NULL,
  `permittivity_max` decimal(7, 3) NULL DEFAULT NULL,
  `conductivity_min` decimal(9, 4) NULL DEFAULT NULL,
  `conductivity_max` decimal(9, 4) NULL DEFAULT NULL,
  `peplinski_dimension` decimal(6, 3) NULL DEFAULT NULL,
  `area_description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `signal_behavior` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL,
  `semantic_usage` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL,
  `created_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  `updated_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `uk_area_domain_code`(`code` ASC) USING BTREE,
  INDEX `idx_area_domain_type`(`domain_type` ASC) USING BTREE,
  INDEX `idx_area_domain_active_order`(`is_active` ASC, `display_order` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 14 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
