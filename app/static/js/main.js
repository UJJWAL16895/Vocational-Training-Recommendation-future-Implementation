// Main JavaScript for Vocational Training Recommendation System

document.addEventListener('DOMContentLoaded', function() {
    // Initialize any components that need JavaScript functionality
    initializeFormValidation();
    initializeTooltips();
    initializeResponsiveNav();
});

/**
 * Form validation for the recommendation form
 */
function initializeFormValidation() {
    const recommendationForm = document.querySelector('.recommendation-form');
    
    if (recommendationForm) {
        recommendationForm.addEventListener('submit', function(event) {
            let isValid = true;
            
            // Validate interests (at least one should be selected)
            const interestCheckboxes = document.querySelectorAll('input[name="interests"]');
            let interestSelected = false;
            
            interestCheckboxes.forEach(function(checkbox) {
                if (checkbox.checked) {
                    interestSelected = true;
                }
            });
            
            if (!interestSelected) {
                isValid = false;
                showError('Please select at least one interest');
            }
            
            // Validate required select fields
            const requiredSelects = recommendationForm.querySelectorAll('select[required]');
            
            requiredSelects.forEach(function(select) {
                if (select.value === '') {
                    isValid = false;
                    select.classList.add('error');
                    showError(`Please select an option for ${select.previousElementSibling.textContent}`);
                } else {
                    select.classList.remove('error');
                }
            });
            
            if (!isValid) {
                event.preventDefault();
            }
        });
    }
}

/**
 * Show error message to the user
 */
function showError(message) {
    // Check if error container already exists
    let errorContainer = document.querySelector('.form-error-container');
    
    if (!errorContainer) {
        // Create error container if it doesn't exist
        errorContainer = document.createElement('div');
        errorContainer.className = 'form-error-container';
        
        // Add styles to the error container
        errorContainer.style.backgroundColor = '#f8d7da';
        errorContainer.style.color = '#721c24';
        errorContainer.style.padding = '1rem';
        errorContainer.style.marginBottom = '1rem';
        errorContainer.style.borderRadius = '4px';
        errorContainer.style.fontWeight = '500';
        
        // Insert at the top of the form
        const form = document.querySelector('.recommendation-form');
        form.insertBefore(errorContainer, form.firstChild);
    }
    
    // Add the error message
    const errorMessage = document.createElement('p');
    errorMessage.textContent = message;
    errorMessage.style.margin = '0.5rem 0';
    
    errorContainer.appendChild(errorMessage);
    
    // Scroll to the error container
    errorContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    // Remove error after 5 seconds
    setTimeout(function() {
        errorContainer.removeChild(errorMessage);
        
        // Remove the container if no more errors
        if (errorContainer.children.length === 0) {
            errorContainer.remove();
        }
    }, 5000);
}

/**
 * Initialize tooltips for additional information
 */
function initializeTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    
    tooltipElements.forEach(function(element) {
        // Add tooltip styling
        element.style.position = 'relative';
        element.style.cursor = 'help';
        element.style.borderBottom = '1px dotted #7f8c8d';
        
        // Create tooltip
        element.addEventListener('mouseenter', function() {
            const tooltipText = this.getAttribute('data-tooltip');
            
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = tooltipText;
            
            // Style the tooltip
            tooltip.style.position = 'absolute';
            tooltip.style.bottom = '100%';
            tooltip.style.left = '50%';
            tooltip.style.transform = 'translateX(-50%)';
            tooltip.style.padding = '0.5rem';
            tooltip.style.backgroundColor = '#2c3e50';
            tooltip.style.color = '#fff';
            tooltip.style.borderRadius = '4px';
            tooltip.style.fontSize = '0.9rem';
            tooltip.style.zIndex = '100';
            tooltip.style.width = 'max-content';
            tooltip.style.maxWidth = '250px';
            tooltip.style.textAlign = 'center';
            
            // Add arrow
            tooltip.style.setProperty('--tooltip-arrow', '"\'"');
            tooltip.style.setProperty('--tooltip-arrow-size', '6px');
            
            this.appendChild(tooltip);
        });
        
        // Remove tooltip
        element.addEventListener('mouseleave', function() {
            const tooltip = this.querySelector('.tooltip');
            if (tooltip) {
                tooltip.remove();
            }
        });
    });
}

/**
 * Initialize responsive navigation for mobile devices
 */
function initializeResponsiveNav() {
    const header = document.querySelector('header');
    
    if (header) {
        // Check if mobile nav already exists
        if (!document.querySelector('.mobile-nav-toggle')) {
            // Create mobile nav toggle button
            const mobileNavToggle = document.createElement('button');
            mobileNavToggle.className = 'mobile-nav-toggle';
            mobileNavToggle.setAttribute('aria-label', 'Toggle navigation');
            mobileNavToggle.innerHTML = '<span></span><span></span><span></span>';
            
            // Style the toggle button
            mobileNavToggle.style.display = 'none';
            mobileNavToggle.style.flexDirection = 'column';
            mobileNavToggle.style.justifyContent = 'space-between';
            mobileNavToggle.style.width = '30px';
            mobileNavToggle.style.height = '21px';
            mobileNavToggle.style.background = 'none';
            mobileNavToggle.style.border = 'none';
            mobileNavToggle.style.cursor = 'pointer';
            mobileNavToggle.style.padding = '0';
            
            // Style the toggle button spans
            const spans = mobileNavToggle.querySelectorAll('span');
            spans.forEach(function(span) {
                span.style.display = 'block';
                span.style.height = '3px';
                span.style.width = '100%';
                span.style.backgroundColor = '#fff';
                span.style.borderRadius = '3px';
                span.style.transition = 'all 0.3s ease';
            });
            
            // Add toggle button to header
            const headerContainer = header.querySelector('.container');
            headerContainer.style.display = 'flex';
            headerContainer.style.justifyContent = 'space-between';
            headerContainer.style.alignItems = 'center';
            headerContainer.appendChild(mobileNavToggle);
            
            // Toggle navigation on button click
            mobileNavToggle.addEventListener('click', function() {
                const nav = header.querySelector('nav');
                nav.classList.toggle('active');
                this.classList.toggle('active');
                
                // Animate the toggle button
                if (this.classList.contains('active')) {
                    spans[0].style.transform = 'rotate(45deg) translate(5px, 6px)';
                    spans[1].style.opacity = '0';
                    spans[2].style.transform = 'rotate(-45deg) translate(5px, -6px)';
                } else {
                    spans[0].style.transform = 'none';
                    spans[1].style.opacity = '1';
                    spans[2].style.transform = 'none';
                }
            });
            
            // Add media query for mobile navigation
            const mediaQuery = window.matchMedia('(max-width: 768px)');
            
            function handleMobileNav(e) {
                const nav = header.querySelector('nav');
                
                if (e.matches) {
                    // Mobile view
                    mobileNavToggle.style.display = 'flex';
                    nav.style.display = 'none';
                    nav.style.position = 'absolute';
                    nav.style.top = '100%';
                    nav.style.left = '0';
                    nav.style.width = '100%';
                    nav.style.backgroundColor = '#2c3e50';
                    nav.style.padding = '1rem';
                    nav.style.boxShadow = '0 5px 10px rgba(0, 0, 0, 0.1)';
                    nav.style.zIndex = '100';
                    
                    // Style the nav when active
                    const navStyle = document.createElement('style');
                    navStyle.textContent = `
                        header nav.active {
                            display: block !important;
                        }
                        
                        header nav ul {
                            flex-direction: column;
                        }
                        
                        header nav ul li {
                            margin-right: 0;