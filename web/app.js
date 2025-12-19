// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 The ALICE Authors

/**
 * ALICE Web Management Interface
 * Shared JavaScript utilities and API client
 */

const API_BASE = '';  // Same origin

/**
 * Toast notification system
 */
const Toast = {
    container: null,
    
    init() {
        this.container = document.createElement('div');
        this.container.className = 'toast-container';
        document.body.appendChild(this.container);
    },
    
    show(message, type = 'info', duration = 4000) {
        if (!this.container) this.init();
        
        const icons = {
            success: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>',
            error: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>',
            warning: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>',
            info: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>'
        };
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <span class="toast-icon">${icons[type] || icons.info}</span>
            <span class="toast-message">${message}</span>
            <button class="toast-close" onclick="this.parentElement.remove()">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        `;
        this.container.appendChild(toast);
        
        setTimeout(() => {
            toast.classList.add('hiding');
            setTimeout(() => toast.remove(), 300);
        }, duration);
    },
    
    success(message) { this.show(message, 'success'); },
    error(message) { this.show(message, 'error', 6000); },
    warning(message) { this.show(message, 'warning'); },
    info(message) { this.show(message, 'info'); }
};

/**
 * API Client for ALICE endpoints
 */
const API = {
    async fetch(endpoint, options = {}) {
        try {
            // Automatically include API key if stored
            const apiKey = localStorage.getItem('alice-admin-key');
            const headers = {
                'Content-Type': 'application/json',
                ...options.headers
            };
            
            // Add API key headers if available and not already set
            // Send both X-Api-Key (for regular auth) and X-Admin-Key (for admin endpoints)
            if (apiKey) {
                if (!headers['X-Api-Key']) {
                    headers['X-Api-Key'] = apiKey;
                }
                if (!headers['X-Admin-Key']) {
                    headers['X-Admin-Key'] = apiKey;
                }
            }
            
            // Build fetch options, excluding headers from options spread (we handle headers separately)
            const { headers: _, ...restOptions } = options;
            const response = await fetch(`${API_BASE}${endpoint}`, {
                headers,
                ...restOptions
            });
            
            if (!response.ok) {
                let errorMsg = `HTTP ${response.status}`;
                try {
                    const errorData = await response.json();
                    // Handle different error response formats
                    if (errorData.error?.message) {
                        errorMsg = errorData.error.message;
                    } else if (errorData.detail) {
                        // FastAPI validation errors return detail as array
                        if (Array.isArray(errorData.detail)) {
                            errorMsg = errorData.detail.map(e => e.msg || String(e)).join(', ');
                        } else if (typeof errorData.detail === 'string') {
                            errorMsg = errorData.detail;
                        } else {
                            errorMsg = JSON.stringify(errorData.detail);
                        }
                    } else if (errorData.message) {
                        errorMsg = typeof errorData.message === 'string' ? errorData.message : JSON.stringify(errorData.message);
                    }
                } catch (e) {
                    // Response might not be JSON, try text
                    try {
                        errorMsg = await response.text() || errorMsg;
                    } catch (e2) {
                        // Keep default error message
                    }
                }
                throw new Error(errorMsg);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`API Error [${endpoint}]:`, error);
            // Show toast for network errors
            if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
                if (Toast && Toast.error) {
                    Toast.error('Network error: Cannot reach server');
                }
            }
            throw error;
        }
    },
    
    // Health & Status
    async getHealth() {
        return this.fetch('/health');
    },
    
    async getMetrics() {
        return this.fetch('/metrics');
    },
    
    // Models
    async listModels() {
        return this.fetch('/v1/models');
    },
    
    async refreshModels() {
        return this.fetch('/v1/models/refresh', { method: 'POST' });
    },
    
    // LoRAs
    async listLoras() {
        return this.fetch('/v1/loras');
    },
    
    // Generation
    async generate(model, prompt, options = {}) {
        // Build sam_config with only defined values - let server use its defaults for undefined
        const sam_config = {};
        
        // Only include values that are explicitly set (not undefined/null/empty)
        if (options.negativePrompt !== undefined && options.negativePrompt !== '') {
            sam_config.negative_prompt = options.negativePrompt;
        }
        if (options.steps !== undefined && !isNaN(options.steps)) {
            sam_config.steps = options.steps;
        }
        if (options.guidanceScale !== undefined && !isNaN(options.guidanceScale)) {
            sam_config.guidance_scale = options.guidanceScale;
        }
        if (options.width !== undefined && !isNaN(options.width)) {
            sam_config.width = options.width;
        }
        if (options.height !== undefined && !isNaN(options.height)) {
            sam_config.height = options.height;
        }
        if (options.seed !== undefined && options.seed !== null && !isNaN(options.seed)) {
            sam_config.seed = options.seed;
        }
        if (options.scheduler !== undefined && options.scheduler !== '') {
            sam_config.scheduler = options.scheduler;
        }
        if (options.numImages !== undefined && !isNaN(options.numImages) && options.numImages > 0) {
            sam_config.num_images = options.numImages;
        }
        if (options.loraPaths && options.loraPaths.length > 0) {
            sam_config.lora_paths = options.loraPaths;
        }
        if (options.loraScales && options.loraScales.length > 0) {
            sam_config.lora_scales = options.loraScales;
        }
        
        const request = {
            model: model,
            messages: [{ role: 'user', content: prompt }],
            sam_config: sam_config
        };
        
        return this.fetch('/v1/chat/completions', {
            method: 'POST',
            body: JSON.stringify(request)
        });
    },
    
    // Authentication
    async getCurrentUser() {
        return this.fetch('/v1/auth/me');
    },
    
    logout() {
        localStorage.removeItem('alice-admin-key');
        window.location.href = '/web/login.html';
    },
    
    isLoggedIn() {
        return !!localStorage.getItem('alice-admin-key');
    }
};

/**
 * UI Utilities
 */
const UI = {
    // Update element text content
    setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    },
    
    // Show/hide element
    show(id) {
        const el = document.getElementById(id);
        if (el) el.style.display = '';
    },
    
    hide(id) {
        const el = document.getElementById(id);
        if (el) el.style.display = 'none';
    },
    
    // Set element class
    setClass(id, className, condition) {
        const el = document.getElementById(id);
        if (el) {
            if (condition) {
                el.classList.add(className);
            } else {
                el.classList.remove(className);
            }
        }
    },
    
    // Format bytes to human readable
    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },
    
    // Format timestamp to local time
    formatTime(timestamp) {
        return new Date(timestamp * 1000).toLocaleString();
    },
    
    // Format duration in seconds
    formatDuration(seconds) {
        if (seconds < 60) return `${seconds.toFixed(1)}s`;
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}m ${secs}s`;
    },
    
    // Debounce function calls
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
};

/**
 * Status polling manager
 */
const StatusPoller = {
    interval: null,
    callbacks: [],
    errorShown: false,
    
    start(intervalMs = 5000) {
        this.stop();
        this.errorShown = false;
        this.poll(); // Initial poll
        this.interval = setInterval(() => this.poll(), intervalMs);
    },
    
    stop() {
        if (this.interval) {
            clearInterval(this.interval);
            this.interval = null;
        }
    },
    
    onUpdate(callback) {
        this.callbacks.push(callback);
    },
    
    async poll() {
        try {
            const [health, metrics] = await Promise.all([
                API.getHealth(),
                API.getMetrics()
            ]);
            
            this.errorShown = false;
            this.callbacks.forEach(cb => cb({ health, metrics }));
        } catch (error) {
            console.error('Status poll failed:', error);
            // Show error once, not on every poll
            if (!this.errorShown) {
                Toast.error(`Cannot connect to ALICE server`);
                this.errorShown = true;
            }
            this.callbacks.forEach(cb => cb({ error }));
        }
    }
};

/**
 * Model selector component
 */
class ModelSelector {
    constructor(selectId) {
        this.select = document.getElementById(selectId);
        this.models = [];
    }
    
    async load() {
        try {
            const response = await API.listModels();
            this.models = response.data || [];
            this.render();
        } catch (error) {
            console.error('Failed to load models:', error);
            Toast.error(`Failed to load models: ${error.message}`);
            this.render(); // Render empty state even on error
        }
    }
    
    render() {
        if (!this.select) return;
        
        this.select.innerHTML = '';
        
        if (this.models.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No models available';
            option.disabled = true;
            option.selected = true;
            this.select.appendChild(option);
            return;
        }
        
        this.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = model.id;
            this.select.appendChild(option);
        });
    }
    
    getValue() {
        return this.select ? this.select.value : '';
    }
}

/**
 * Image gallery component
 */
class ImageGallery {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.images = [];
        this.onClick = null; // Callback for image click
    }
    
    add(imageUrl, metadata = {}) {
        this.images.unshift({ url: imageUrl, metadata, timestamp: Date.now() });
        // Keep only last 20 images
        if (this.images.length > 20) {
            this.images = this.images.slice(0, 20);
        }
        this.render();
    }
    
    render() {
        if (!this.container) return;
        
        if (this.images.length === 0) {
            this.container.innerHTML = `
                <div class="empty-state">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <p class="title">No images yet</p>
                    <p>Generated images will appear here</p>
                </div>
            `;
            return;
        }
        
        this.container.innerHTML = this.images.map((img, index) => `
            <div class="gallery-item" data-index="${index}">
                <img src="${img.url}" alt="Generated image" loading="lazy">
                <div class="overlay">
                    ${img.metadata.prompt ? this.truncate(img.metadata.prompt, 50) : 'Generated image'}
                </div>
            </div>
        `).join('');
        
        // Add click handlers
        this.container.querySelectorAll('.gallery-item').forEach(item => {
            item.addEventListener('click', () => {
                const index = parseInt(item.dataset.index);
                const image = this.images[index];
                if (this.onClick && image) {
                    this.onClick(image.url, image.metadata);
                }
            });
        });
    }
    
    truncate(str, len) {
        return str.length > len ? str.substring(0, len) + '...' : str;
    }
}

// Add scroll listener for nav shadow
document.addEventListener('scroll', () => {
    const nav = document.querySelector('nav');
    if (nav) {
        nav.classList.toggle('scrolled', window.scrollY > 10);
    }
});

/**
 * Authentication guard for pages.
 * Call this at the start of page load to redirect unauthenticated users.
 * 
 * @param {Object} options
 * @param {boolean} options.required - If true, always require auth. If false, check server config.
 * @param {boolean} options.adminOnly - If true, redirect non-admins to generate page.
 * @returns {Promise<{authenticated: boolean, apiKey: string|null, isAdmin: boolean}>}
 */
async function requireAuth(options = {}) {
    const currentPath = window.location.pathname;
    
    // Skip auth check for login page itself
    if (currentPath.includes('login.html')) {
        return { authenticated: false, apiKey: null, isAdmin: false };
    }
    
    try {
        // Check if server requires auth
        const registrationMode = await API.fetch('/v1/auth/registration-mode');
        const serverRequiresAuth = registrationMode.require_auth;
        
        // Check for stored key
        const apiKey = localStorage.getItem('alice-admin-key');
        console.log('[requireAuth] API key from localStorage:', apiKey ? 'found' : 'not found');
        
        // If user has an API key, verify it and get their info
        if (apiKey) {
            try {
                console.log('[requireAuth] Verifying key...');
                const userInfo = await API.fetch('/v1/auth/me', {
                    headers: { 'X-Api-Key': apiKey }
                });
                console.log('[requireAuth] userInfo:', JSON.stringify(userInfo));
                
                if (userInfo.authenticated) {
                    // Valid key - update nav and return user info
                    console.log('[requireAuth] Authenticated! is_admin:', userInfo.is_admin);
                    updateNavVisibility(true, userInfo.is_admin);
                    
                    // Check admin access for admin-only pages
                    if (options.adminOnly && !userInfo.is_admin) {
                        Toast.error('Admin access required');
                        window.location.href = '/web/generate.html';
                        return { authenticated: true, apiKey, isAdmin: false };
                    }
                    
                    return { authenticated: true, apiKey, isAdmin: userInfo.is_admin };
                } else {
                    console.log('[requireAuth] userInfo.authenticated is false');
                }
            } catch (e) {
                // Key verification failed - clear it
                console.log('[requireAuth] Key verification error:', e);
                localStorage.removeItem('alice-admin-key');
                clearSessionCookie();
            }
        }
        
        // No valid key - check if auth is required
        if (!serverRequiresAuth && !options.required) {
            // Server doesn't require auth - show nav links for anonymous user
            updateNavVisibility(true, false);
            return { authenticated: false, apiKey: null, isAdmin: false };
        }
        
        // Server requires auth and we don't have a valid key
        updateNavVisibility(false, false);
        redirectToLogin(currentPath);
        return { authenticated: false, apiKey: null, isAdmin: false };
    } catch (error) {
        // If we can't check config, assume no auth required
        console.warn('Could not check auth config:', error);
        return { authenticated: false, apiKey: null, isAdmin: false };
    }
}

/**
 * Update navigation visibility based on authentication status
 * @param {boolean} isAuthenticated - Whether user is authenticated
 * @param {boolean} isAdmin - Whether user is admin
 */
function updateNavVisibility(isAuthenticated, isAdmin) {
    // Set body class for CSS-based visibility (belt and suspenders approach)
    if (isAdmin) {
        document.body.classList.add('is-admin');
    } else {
        document.body.classList.remove('is-admin');
    }
    
    if (isAuthenticated) {
        document.body.classList.add('is-authenticated');
    } else {
        document.body.classList.remove('is-authenticated');
    }
    
    // Also set inline styles for immediate effect
    document.querySelectorAll('.admin-only').forEach(el => {
        el.style.display = isAdmin ? '' : 'none';
    });
    
    // Hide/show logout link (only when authenticated)
    document.querySelectorAll('.logout-link').forEach(el => {
        el.style.display = isAuthenticated ? '' : 'none';
    });
    
    // Hide/show login link (only when NOT authenticated)
    document.querySelectorAll('.login-link').forEach(el => {
        el.style.display = isAuthenticated ? 'none' : '';
    });
}

/**
 * Redirect to login page with return URL
 */
function redirectToLogin(returnPath) {
    const loginUrl = `/web/login.html?return=${encodeURIComponent(returnPath)}`;
    window.location.href = loginUrl;
}

/**
 * Set the session cookie for server-side auth
 */
function setSessionCookie(apiKey) {
    // Set cookie with SameSite=Lax to allow cookies on top-level GET navigations (redirects)
    // Cookie expires when browser closes (session cookie)
    document.cookie = `alice_session=${apiKey}; path=/; SameSite=Lax`;
}

/**
 * Clear the session cookie
 */
function clearSessionCookie() {
    document.cookie = 'alice_session=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT; SameSite=Lax';
}

/**
 * Logout - clear stored credentials and session cookie, redirect to login
 */
function logout() {
    localStorage.removeItem('alice-admin-key');
    clearSessionCookie();
    window.location.href = '/web/login.html';
}

/**
 * Session timeout manager
 * Tracks user activity and auto-logs out after inactivity
 */
const SessionManager = {
    timeoutId: null,
    warningTimeoutId: null,
    lastActivity: Date.now(),
    timeoutSeconds: 900, // Default 15 minutes, updated from server
    warningSeconds: 60, // Show warning 1 minute before timeout
    isInitialized: false,
    warningShown: false,
    
    /**
     * Initialize session manager.
     * Fetches timeout settings from server and starts monitoring.
     */
    async init() {
        if (this.isInitialized) return;
        this.isInitialized = true;
        
        // Only monitor if user is logged in
        const apiKey = localStorage.getItem('alice-admin-key');
        if (!apiKey) return;
        
        // Get timeout settings from server
        try {
            const userInfo = await API.fetch('/v1/auth/me', {
                headers: { 'X-Api-Key': apiKey }
            });
            
            if (userInfo.session_timeout_seconds) {
                this.timeoutSeconds = userInfo.session_timeout_seconds;
            }
            if (userInfo.inactivity_timeout_seconds) {
                this.timeoutSeconds = Math.min(this.timeoutSeconds, userInfo.inactivity_timeout_seconds);
            }
        } catch (e) {
            console.warn('Could not fetch session timeout settings:', e);
        }
        
        // Set up activity listeners
        const activityEvents = ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart', 'click'];
        activityEvents.forEach(event => {
            document.addEventListener(event, () => this.recordActivity(), { passive: true });
        });
        
        // Start monitoring
        this.recordActivity();
        this.startMonitoring();
        
        console.log(`Session manager initialized: timeout=${this.timeoutSeconds}s`);
    },
    
    /**
     * Record user activity and reset timers.
     */
    recordActivity() {
        this.lastActivity = Date.now();
        this.warningShown = false;
        
        // Reset timers
        this.startMonitoring();
    },
    
    /**
     * Start the timeout monitoring.
     */
    startMonitoring() {
        // Clear existing timers
        if (this.timeoutId) clearTimeout(this.timeoutId);
        if (this.warningTimeoutId) clearTimeout(this.warningTimeoutId);
        
        // Set warning timer (1 minute before timeout)
        const warningMs = Math.max(0, (this.timeoutSeconds - this.warningSeconds) * 1000);
        this.warningTimeoutId = setTimeout(() => this.showWarning(), warningMs);
        
        // Set logout timer
        const timeoutMs = this.timeoutSeconds * 1000;
        this.timeoutId = setTimeout(() => this.handleTimeout(), timeoutMs);
    },
    
    /**
     * Show warning before timeout.
     */
    showWarning() {
        if (this.warningShown) return;
        this.warningShown = true;
        
        Toast.warning(`Session expires in ${this.warningSeconds} seconds. Move your mouse to stay logged in.`);
    },
    
    /**
     * Handle session timeout - logout user.
     */
    handleTimeout() {
        const apiKey = localStorage.getItem('alice-admin-key');
        if (!apiKey) return; // Already logged out
        
        // Clear the stored key and session cookie
        localStorage.removeItem('alice-admin-key');
        clearSessionCookie();
        
        // Show message and redirect
        Toast.warning('Session expired due to inactivity. Please log in again.');
        
        setTimeout(() => {
            window.location.href = '/web/login.html?expired=1';
        }, 2000);
    },
    
    /**
     * Stop monitoring (call on logout).
     */
    stop() {
        if (this.timeoutId) clearTimeout(this.timeoutId);
        if (this.warningTimeoutId) clearTimeout(this.warningTimeoutId);
        this.isInitialized = false;
    },
    
    /**
     * Get time remaining until timeout in seconds.
     */
    getTimeRemaining() {
        const elapsed = (Date.now() - this.lastActivity) / 1000;
        return Math.max(0, this.timeoutSeconds - elapsed);
    }
};

// Export for use in other scripts
window.SDAPI = { API, UI, Toast, StatusPoller, ModelSelector, ImageGallery, requireAuth, logout, SessionManager, setSessionCookie, clearSessionCookie, updateNavVisibility };

// Initialize Toast and SessionManager when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Toast
    Toast.init();
    
    // Initialize session manager if user has API key
    const apiKey = localStorage.getItem('alice-admin-key');
    setTimeout(() => {
        if (apiKey) {
            SessionManager.init();
        }
    }, 100);
    
    // Note: Navigation visibility is handled by each page's requireAuth() call
    // to avoid race conditions between app.js and page-specific initialization
});
