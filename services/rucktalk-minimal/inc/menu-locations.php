<?php
/**
 * Menu locations — rucktalk-minimal.
 *
 * Sonaar parent already registers `main-menu` + `responsive-menu` locations,
 * and Mike's existing "Main Menu" (term ID 6) is attached to both. We DO NOT
 * register competing locations — header.php (Task 9) renders the existing
 * menu via `wp_nav_menu( theme_location => 'main-menu' )`.
 *
 * This file exists as a load slot for any future menu work (e.g., a separate
 * footer menu if we decide to add one in a later phase).
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

// Task 6 may add a `footer` location later. For Phase 1A, parent's locations suffice.
