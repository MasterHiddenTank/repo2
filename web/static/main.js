/**
 * Main JavaScript file for SPY Stock Price Prediction web application.
 * This file contains client-side functionality for the web interface.
 */

// Execute when DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Add any additional initialization code here
    console.log('SPY Stock Price Prediction application initialized');
});

/**
 * Format a date object into a readable string
 * 
 * @param {Date} date - Date object to format
 * @returns {string} Formatted date string
 */
function formatDate(date) {
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

/**
 * Format price with dollar sign and 2 decimal places
 * 
 * @param {number} price - Price to format
 * @returns {string} Formatted price string
 */
function formatPrice(price) {
    return '$' + parseFloat(price).toFixed(2);
}

/**
 * Create a nice alert message that automatically disappears
 * 
 * @param {string} message - Message to display
 * @param {string} type - Alert type (success, danger, warning, info)
 * @param {number} duration - Time in milliseconds before alert disappears
 */
function createAlert(message, type, duration) {
    if (type === undefined) type = 'info';
    if (duration === undefined) duration = 5000;
    
    // Create alert element
    var alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-' + type + ' alert-dismissible fade show';
    alertDiv.setAttribute('role', 'alert');
    
    // Add message
    alertDiv.innerHTML = 
        message +
        '<button type="button" class="close" data-dismiss="alert" aria-label="Close">' +
        '<span aria-hidden="true">&times;</span>' +
        '</button>';
    
    // Add to page
    var alertContainer = document.querySelector('.alert-container');
    if (!alertContainer) {
        alertContainer = document.createElement('div');
        alertContainer.className = 'alert-container position-fixed top-0 right-0 p-3';
        alertContainer.style.zIndex = '1050';
        alertContainer.style.right = '0';
        alertContainer.style.top = '70px';
        document.body.appendChild(alertContainer);
    }
    
    alertContainer.appendChild(alertDiv);
    
    // Auto-remove after duration
    if (duration > 0) {
        setTimeout(function() {
            $(alertDiv).alert('close');
        }, duration);
    }
}

// Export functions for use in other scripts
window.appUtils = {
    formatDate: formatDate,
    formatPrice: formatPrice,
    createAlert: createAlert
};
