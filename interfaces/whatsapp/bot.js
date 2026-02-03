/**
 * Alfred WhatsApp Bot
 *
 * Connects Alfred AI to WhatsApp using whatsapp-web.js
 * Scan the QR code with your phone to authenticate.
 */

const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const axios = require('axios');
const fs = require('fs');
const path = require('path');

// Configuration
const ALFRED_API_URL = process.env.ALFRED_API_URL || 'http://localhost:8400';
const ALFRED_API_TOKEN = process.env.ALFRED_API_TOKEN || '';
const ALLOWED_NUMBERS = (process.env.WHATSAPP_ALLOWED_NUMBERS || '').split(',').filter(n => n);
const SESSION_PATH = path.join(__dirname, '.wwebjs_auth');

// Signature appended to all Alfred responses
const SIGNATURE = "\n\n— _Sent via Alfred, Mike Johnson's personal assistant_";

// Create client with local authentication (persists session)
const client = new Client({
    authStrategy: new LocalAuth({
        dataPath: SESSION_PATH
    }),
    puppeteer: {
        headless: true,
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-accelerated-2d-canvas',
            '--no-first-run',
            '--no-zygote',
            '--disable-gpu'
        ]
    }
});

// Store for conversation context (simple in-memory)
const conversations = new Map();

/**
 * Check if a number is authorized
 */
function isAuthorized(number) {
    if (ALLOWED_NUMBERS.length === 0) return true;
    // Clean the number for comparison
    const cleanNumber = number.replace(/[^0-9]/g, '');
    return ALLOWED_NUMBERS.some(allowed => cleanNumber.includes(allowed.replace(/[^0-9]/g, '')));
}

/**
 * Query Alfred's API
 */
async function askAlfred(message, conversationId) {
    try {
        const headers = {
            'Content-Type': 'application/json',
        };

        if (ALFRED_API_TOKEN) {
            headers['Authorization'] = `Bearer ${ALFRED_API_TOKEN}`;
        }

        // Get conversation history
        const history = conversations.get(conversationId) || [];

        const response = await axios.post(`${ALFRED_API_URL}/chat`, {
            message: message,
            conversation_id: conversationId,
            history: history.slice(-10), // Last 10 messages for context
        }, { headers, timeout: 120000 }); // 2 minute timeout

        const reply = response.data.response || response.data.message || "I couldn't process that request.";

        // Update conversation history
        history.push({ role: 'user', content: message });
        history.push({ role: 'assistant', content: reply });
        conversations.set(conversationId, history.slice(-20)); // Keep last 20 messages

        return reply;
    } catch (error) {
        console.error('Alfred API error:', error.message);

        if (error.code === 'ECONNREFUSED') {
            return "I'm having trouble connecting to my brain. Please make sure Alfred is running.";
        }

        if (error.response?.status === 401) {
            return "Authentication error. Please check the API token configuration.";
        }

        return "Sorry, I encountered an error processing your request. Please try again.";
    }
}

/**
 * Get daily briefing from Alfred
 */
async function getBriefing() {
    try {
        const headers = {};
        if (ALFRED_API_TOKEN) {
            headers['Authorization'] = `Bearer ${ALFRED_API_TOKEN}`;
        }

        const response = await axios.get(`${ALFRED_API_URL}/briefing/quick`, { headers, timeout: 60000 });
        return response.data.text || "Couldn't generate briefing.";
    } catch (error) {
        console.error('Briefing error:', error.message);
        return "Sorry, I couldn't generate your briefing right now.";
    }
}

/**
 * Handle incoming messages
 */
async function handleMessage(message) {
    // Debug: log all message properties
    console.log(`[DEBUG] Received message - fromMe: ${message.fromMe}, isStatus: ${message.isStatus}, isGroupMsg: ${message.isGroupMsg}, body length: ${message.body?.length || 0}`);

    // Ignore status updates
    if (message.isStatus) return;

    // For self-messages, only respond if NOT fromMe (incoming from another device)
    // This allows testing from another phone messaging this number
    if (message.fromMe) {
        console.log('[DEBUG] Ignoring self-sent message');
        return;
    }

    // Ignore group messages unless Alfred is mentioned
    if (message.isGroupMsg) {
        const body = message.body || '';
        if (!body.toLowerCase().includes('alfred')) {
            console.log('[DEBUG] Ignoring group message without Alfred mention');
            return;
        }
    }

    const contact = await message.getContact();
    const number = contact.number || 'unknown';
    const name = contact.pushname || contact.name || 'there';

    console.log(`[${new Date().toISOString()}] Message from ${name} (${number}): ${(message.body || '').substring(0, 100)}`);

    // Check authorization
    if (!isAuthorized(number)) {
        console.log(`Unauthorized number: ${number}`);
        await message.reply("Sorry, you're not authorized to use Alfred. Please contact the administrator.");
        return;
    }

    const text = message.body.trim();
    const conversationId = `whatsapp_${number}`;

    // Handle commands
    if (text.toLowerCase() === '/start' || text.toLowerCase() === 'hi alfred' || text.toLowerCase() === 'hey alfred') {
        await message.reply(
            `Hello ${name}! I'm Alfred, Mike's AI assistant.\n\n` +
            `You can ask me anything, or use these commands:\n` +
            `/briefing - Get your daily briefing\n` +
            `/email - Check your emails\n` +
            `/calendar - Check today's schedule\n` +
            `/help - Show all commands\n\n` +
            `Just send me a message to get started!` + SIGNATURE
        );
        return;
    }

    if (text.toLowerCase() === '/help') {
        await message.reply(
            `*Alfred Commands:*\n\n` +
            `/briefing - Get your daily briefing\n` +
            `/email - Check your emails\n` +
            `/calendar - Today's schedule\n` +
            `/clear - Clear conversation history\n` +
            `/status - Check Alfred's status\n\n` +
            `Or just send me any message and I'll help!` + SIGNATURE
        );
        return;
    }

    if (text.toLowerCase() === '/briefing') {
        await message.reply("Generating your briefing...");
        const briefing = await getBriefing();
        await message.reply(briefing + SIGNATURE);
        return;
    }

    if (text.toLowerCase() === '/clear') {
        conversations.delete(conversationId);
        await message.reply("Conversation history cleared. Let's start fresh!" + SIGNATURE);
        return;
    }

    if (text.toLowerCase() === '/status') {
        await message.reply(
            `*Alfred Status*\n\n` +
            `Status: Online\n` +
            `Interface: WhatsApp\n` +
            `Ready to assist!` + SIGNATURE
        );
        return;
    }

    // Regular message - send to Alfred
    // Show typing indicator (WhatsApp doesn't have a direct API for this, but we can simulate)
    const chat = await message.getChat();
    await chat.sendStateTyping();

    const reply = await askAlfred(text, conversationId);

    // Add signature to the reply
    const replyWithSig = reply + SIGNATURE;

    // WhatsApp has a character limit, split long messages
    if (replyWithSig.length > 4000) {
        const chunks = reply.match(/.{1,3900}/gs) || [reply];
        for (let i = 0; i < chunks.length; i++) {
            // Add signature only to the last chunk
            const chunkText = i === chunks.length - 1 ? chunks[i] + SIGNATURE : chunks[i];
            await message.reply(chunkText);
        }
    } else {
        await message.reply(replyWithSig);
    }
}

// Event handlers
client.on('qr', (qr) => {
    console.log('\n========================================');
    console.log('Scan this QR code with WhatsApp:');
    console.log('========================================\n');
    qrcode.generate(qr, { small: true });
    console.log('\nOpen WhatsApp > Settings > Linked Devices > Link a Device');
    console.log('========================================\n');
});

client.on('authenticated', () => {
    console.log('WhatsApp authenticated successfully!');
});

client.on('auth_failure', (msg) => {
    console.error('WhatsApp authentication failed:', msg);
});

client.on('ready', () => {
    console.log('\n========================================');
    console.log('Alfred WhatsApp Bot is ready!');
    console.log(`API URL: ${ALFRED_API_URL}`);
    console.log(`Allowed numbers: ${ALLOWED_NUMBERS.length > 0 ? ALLOWED_NUMBERS.join(', ') : 'All'}`);
    console.log('========================================\n');
});

client.on('message', handleMessage);

client.on('disconnected', (reason) => {
    console.log('WhatsApp client disconnected:', reason);
    // Attempt to reconnect
    setTimeout(() => {
        console.log('Attempting to reconnect...');
        client.initialize();
    }, 5000);
});

// Graceful shutdown
process.on('SIGINT', async () => {
    console.log('\nShutting down...');
    await client.destroy();
    process.exit(0);
});

process.on('SIGTERM', async () => {
    console.log('\nShutting down...');
    await client.destroy();
    process.exit(0);
});

// Start the client
console.log('Starting Alfred WhatsApp Bot...');
console.log('This may take a moment on first run...\n');
client.initialize();
