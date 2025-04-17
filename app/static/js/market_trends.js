// Market Trends Visualization for Vocational Training Recommendation System

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the market trends visualizations
    initializeMarketTrendsCharts();
    
    // Add event listener for region selector
    const regionSelect = document.getElementById('region-select');
    if (regionSelect) {
        regionSelect.addEventListener('change', function() {
            updateChartsForRegion(this.value);
        });
    }
});

/**
 * Initialize all charts for market trends visualization
 */
function initializeMarketTrendsCharts() {
    // Show loading indicator
    const chartsContainer = document.querySelector('.charts-container');
    if (chartsContainer) {
        chartsContainer.classList.add('loading');
    }
    
    // Fetch market trends data from the API
    fetch('/api/market-trends')
        .then(response => response.json())
        .then(data => {
            // Initialize charts with the data
            createIndustryDemandChart(data);
            createRegionalComparisonChart(data);
            createSkillsGapChart(data);
            createJobGrowthChart(data);
            
            // Remove loading indicator with a slight delay for better UX
            setTimeout(() => {
                if (chartsContainer) {
                    chartsContainer.classList.remove('loading');
                    
                    // Add fade-in animation to chart cards
                    const chartCards = document.querySelectorAll('.chart-card');
                    chartCards.forEach((card, index) => {
                        card.style.opacity = '0';
                        card.style.transform = 'translateY(20px)';
                        card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                        
                        setTimeout(() => {
                            card.style.opacity = '1';
                            card.style.transform = 'translateY(0)';
                        }, 100 * index);
                    });
                }
            }, 500);
        })
        .catch(error => {
            console.error('Error fetching market trends data:', error);
            // Display error message to user
            if (chartsContainer) {
                chartsContainer.classList.remove('loading');
                chartsContainer.innerHTML = `
                    <div class="error-message">
                        <p>Unable to load market trends data. Please try again later.</p>
                    </div>
                `;
            }
        });
}

/**
 * Update all charts based on selected region
 */
function updateChartsForRegion(region) {
    // Show loading indicator
    const chartsContainer = document.querySelector('.charts-container');
    if (chartsContainer) {
        chartsContainer.classList.add('loading');
        
        // Add transition effect to chart cards
        const chartCards = document.querySelectorAll('.chart-card');
        chartCards.forEach(card => {
            card.style.opacity = '0.6';
            card.style.transform = 'scale(0.98)';
            card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        });
    }
    
    // Update region name in UI
    const regionTitle = document.querySelector('.region-title');
    if (regionTitle) {
        const regionName = region.charAt(0).toUpperCase() + region.slice(1);
        regionTitle.textContent = region === 'all' ? 'All Regions' : regionName;
    }
    
    // Fetch data for the selected region
    fetch(`/api/market-trends?region=${region}`)
        .then(response => response.json())
        .then(data => {
            // Update charts with new data
            updateIndustryDemandChart(data);
            updateRegionalComparisonChart(data);
            updateSkillsGapChart(data);
            updateJobGrowthChart(data);
            
            // Remove loading indicator with a slight delay for better UX
            setTimeout(() => {
                if (chartsContainer) {
                    chartsContainer.classList.remove('loading');
                    
                    // Restore chart cards with animation
                    const chartCards = document.querySelectorAll('.chart-card');
                    chartCards.forEach((card, index) => {
                        setTimeout(() => {
                            card.style.opacity = '1';
                            card.style.transform = 'scale(1)';
                        }, 100 * index);
                    });
                }
            }, 300);
        })
        .catch(error => {
            console.error('Error updating market trends data:', error);
            // Display error message
            if (chartsContainer) {
                chartsContainer.classList.remove('loading');
                chartsContainer.innerHTML = `
                    <div class="error-message">
                        <p>Unable to update market trends data. Please try again later.</p>
                    </div>
                `;
            }
        });
}

/**
 * Create chart showing industry demand by region
 */
function createIndustryDemandChart(data) {
    const ctx = document.getElementById('industry-demand-chart');
    if (!ctx) return;
    
    // Extract data for the chart
    const labels = Object.keys(data.industry_demand);
    const values = Object.values(data.industry_demand);
    
    // Create the chart
    window.industryDemandChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Industry Demand Score',
                data: values,
                backgroundColor: [
                    'rgba(54, 162, 235, 0.7)',
                    'rgba(75, 192, 192, 0.7)',
                    'rgba(255, 206, 86, 0.7)',
                    'rgba(153, 102, 255, 0.7)',
                    'rgba(255, 159, 64, 0.7)',
                    'rgba(255, 99, 132, 0.7)'
                ],
                borderColor: [
                    'rgba(54, 162, 235, 1)',
                    'rgba(75, 192, 192, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(153, 102, 255, 1)',
                    'rgba(255, 159, 64, 1)',
                    'rgba(255, 99, 132, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Industry Demand by Sector',
                    font: {
                        size: 16
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Demand Score: ${(context.raw * 100).toFixed(1)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100) + '%';
                        }
                    }
                }
            }
        }
    });
}

/**
 * Update industry demand chart with new data
 */
function updateIndustryDemandChart(data) {
    if (!window.industryDemandChart) return;
    
    const labels = Object.keys(data.industry_demand);
    const values = Object.values(data.industry_demand);
    
    window.industryDemandChart.data.labels = labels;
    window.industryDemandChart.data.datasets[0].data = values;
    window.industryDemandChart.update();
}

/**
 * Create chart comparing regions
 */
function createRegionalComparisonChart(data) {
    const ctx = document.getElementById('regional-comparison-chart');
    if (!ctx) return;
    
    // Extract data for the chart
    const regions = Object.keys(data.regional_comparison);
    const datasets = [];
    
    // Get all industries
    const industries = Object.keys(data.regional_comparison[regions[0]]);
    
    // Create datasets for each industry
    industries.forEach((industry, index) => {
        const industryData = regions.map(region => data.regional_comparison[region][industry]);
        
        datasets.push({
            label: industry.charAt(0).toUpperCase() + industry.slice(1),
            data: industryData,
            backgroundColor: getColorForIndex(index, 0.7),
            borderColor: getColorForIndex(index, 1),
            borderWidth: 1
        });
    });
    
    // Create the chart
    window.regionalComparisonChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: regions.map(region => region.charAt(0).toUpperCase() + region.slice(1)),
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Regional Industry Comparison',
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                r: {
                    min: 0,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100) + '%';
                        }
                    }
                }
            }
        }
    });
}

/**
 * Update regional comparison chart with new data
 */
function updateRegionalComparisonChart(data) {
    if (!window.regionalComparisonChart) return;
    
    const regions = Object.keys(data.regional_comparison);
    const industries = Object.keys(data.regional_comparison[regions[0]]);
    
    // Update datasets
    industries.forEach((industry, index) => {
        const industryData = regions.map(region => data.regional_comparison[region][industry]);
        
        if (index < window.regionalComparisonChart.data.datasets.length) {
            window.regionalComparisonChart.data.datasets[index].data = industryData;
        }
    });
    
    window.regionalComparisonChart.update();
}

/**
 * Create chart showing skills gap
 */
function createSkillsGapChart(data) {
    const ctx = document.getElementById('skills-gap-chart');
    if (!ctx) return;
    
    // Extract data for the chart
    const skills = Object.keys(data.skills_gap);
    const demandValues = skills.map(skill => data.skills_gap[skill].demand);
    const supplyValues = skills.map(skill => data.skills_gap[skill].supply);
    
    // Create the chart
    window.skillsGapChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: skills.map(skill => skill.charAt(0).toUpperCase() + skill.slice(1)),
            datasets: [
                {
                    label: 'Demand',
                    data: demandValues,
                    backgroundColor: 'rgba(54, 162, 235, 0.7)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Supply',
                    data: supplyValues,
                    backgroundColor: 'rgba(255, 99, 132, 0.7)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Skills Gap Analysis',
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100) + '%';
                        }
                    }
                }
            }
        }
    });
}

/**
 * Update skills gap chart with new data
 */
function updateSkillsGapChart(data) {
    if (!window.skillsGapChart) return;
    
    const skills = Object.keys(data.skills_gap);
    const demandValues = skills.map(skill => data.skills_gap[skill].demand);
    const supplyValues = skills.map(skill => data.skills_gap[skill].supply);
    
    window.skillsGapChart.data.labels = skills.map(skill => skill.charAt(0).toUpperCase() + skill.slice(1));
    window.skillsGapChart.data.datasets[0].data = demandValues;
    window.skillsGapChart.data.datasets[1].data = supplyValues;
    window.skillsGapChart.update();
}

/**
 * Create chart showing job growth trends
 */
function createJobGrowthChart(data) {
    const ctx = document.getElementById('job-growth-chart');
    if (!ctx) return;
    
    // Extract data for the chart
    const years = data.job_growth.years;
    const industries = Object.keys(data.job_growth.industries);
    
    const datasets = industries.map((industry, index) => {
        return {
            label: industry.charAt(0).toUpperCase() + industry.slice(1),
            data: data.job_growth.industries[industry],
            borderColor: getColorForIndex(index, 1),
            backgroundColor: getColorForIndex(index, 0.1),
            borderWidth: 2,
            fill: false,
            tension: 0.4
        };
    });
    
    // Create the chart
    window.jobGrowthChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: years,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Job Growth Trends (5-Year Projection)',
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Growth Rate'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
}

/**
 * Update job growth chart with new data
 */
function updateJobGrowthChart(data) {
    if (!window.jobGrowthChart) return;
    
    const years = data.job_growth.years;
    const industries = Object.keys(data.job_growth.industries);
    
    window.jobGrowthChart.data.labels = years;
    
    industries.forEach((industry, index) => {
        if (index < window.jobGrowthChart.data.datasets.length) {
            window.jobGrowthChart.data.datasets[index].data = data.job_growth.industries[industry];
        }
    });
    
    window.jobGrowthChart.update();
}

/**
 * Helper function to get color for chart elements
 */
function getColorForIndex(index, alpha) {
    const colors = [
        `rgba(54, 162, 235, ${alpha})`,
        `rgba(255, 99, 132, ${alpha})`,
        `rgba(75, 192, 192, ${alpha})`,
        `rgba(255, 206, 86, ${alpha})`,
        `rgba(153, 102, 255, ${alpha})`,
        `rgba(255, 159, 64, ${alpha})`
    ];
    
    return colors[index % colors.length];
}