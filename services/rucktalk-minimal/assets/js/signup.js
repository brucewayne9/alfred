/*
 * rucktalk-minimal — signup form handler.
 *
 * Intercepts every .rt-signup-form submit on the page (event delegation
 * at document level, so dynamically-added forms — popup, late-rendered
 * inline blocks — work without rebinding).
 *
 * Path:
 *   1. Read email + placement from the form
 *   2. POST JSON to RuckTalkSignup.restUrl with x-wp-nonce header
 *   3. Render state into form's .rt-signup-form__status (role="status")
 *
 * No-JS path: the form's native action attribute posts to
 * admin-post.php?action=rt_signup which inc/rest-signup.php handles.
 * Both paths share the same WP handler; this script just upgrades the
 * UX so the user stays on the page.
 *
 * No jQuery. RuckTalkSignup is injected by wp_localize_script in
 * functions.php (restUrl + nonce). If it's missing, we bail silently
 * and the native form-submit fallback takes over.
 */
( function () {
    'use strict';

    if ( ! window.RuckTalkSignup || ! window.RuckTalkSignup.restUrl ) {
        return;
    }

    var SENDING_MSG = 'Sending…';
    var OK_MSG      = '✓ Check your email to confirm. The plan lands the moment you click verify.';
    var ERR_MSG     = 'Something went wrong. Please try again or email mike@rucktalk.com.';
    var NET_MSG     = 'Network error. Please try again.';

    function setStatus( status, text, state ) {
        if ( ! status ) { return; }
        status.textContent = text;
        if ( state ) {
            status.dataset.state = state;
        } else {
            // Remove previous state so transitional "Sending…" doesn't
            // inherit a stale data-state="err".
            delete status.dataset.state;
        }
    }

    document.addEventListener( 'submit', function ( e ) {
        var form = e.target && e.target.closest ? e.target.closest( '.rt-signup-form' ) : null;
        if ( ! form ) {
            return;
        }
        e.preventDefault();

        var emailInput = form.querySelector( 'input[name="email"]' );
        var status     = form.querySelector( '.rt-signup-form__status' );
        var submitBtn  = form.querySelector( 'button[type="submit"]' );

        if ( ! emailInput ) {
            return;
        }
        var email     = ( emailInput.value || '' ).trim();
        var placement = form.dataset.placement || 'unknown';

        if ( '' === email ) {
            setStatus( status, ERR_MSG, 'err' );
            return;
        }

        setStatus( status, SENDING_MSG, '' );
        if ( submitBtn ) {
            submitBtn.disabled = true;
        }

        fetch( window.RuckTalkSignup.restUrl, {
            method:      'POST',
            credentials: 'same-origin',
            headers: {
                'content-type': 'application/json',
                'accept':       'application/json',
                'x-wp-nonce':   window.RuckTalkSignup.nonce || ''
            },
            body: JSON.stringify( { email: email, placement: placement } )
        } ).then( function ( r ) {
            return r.json().then( function ( j ) {
                return { httpOk: r.ok, body: j };
            } ).catch( function () {
                return { httpOk: r.ok, body: {} };
            } );
        } ).then( function ( res ) {
            if ( submitBtn ) { submitBtn.disabled = false; }
            if ( res.httpOk && res.body && res.body.ok ) {
                setStatus( status, OK_MSG, 'ok' );
                emailInput.value = '';
            } else {
                setStatus( status, ERR_MSG, 'err' );
            }
        } ).catch( function () {
            if ( submitBtn ) { submitBtn.disabled = false; }
            setStatus( status, NET_MSG, 'err' );
        } );
    } );
}() );
