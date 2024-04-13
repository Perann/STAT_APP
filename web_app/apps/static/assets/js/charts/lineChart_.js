//
// Sales chart
//

var SalesChart = (function() {

  // Variables

  var $chart = $('#chart-sales-dark');


  // Methods

  function init($chart) {

    var salesChart = new Chart($chart, {
      type: 'line',
      options: {
        scales: {
          yAxes: [{
            gridLines: {
              lineWidth: 1,
              color: Charts.colors.gray[900],
              zeroLineColor: Charts.colors.gray[900]
            },
            ticks: {
              callback: function(value) {
                if (!(value % 10)) {
                  return '$' + value + 'k';
                }
              }
            }
          }]
        },
        tooltips: {
          callbacks: {
            label: function(item, data) {
              var label = data.datasets[item.datasetIndex].label || '';
              var yLabel = item.yLabel;
              var content = '';

              if (data.datasets.length > 1) {
                content += '<span class="popover-body-label mr-auto">' + label + '</span>';
              }

              content += '<span class="popover-body-value">$' + yLabel + 'k</span>';
              return content;
            }
          }
        }
      },
      data: {
        labels: ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        datasets: [{
          label: 'Performance',
          data: [0, 20, 10, 30, 15, 40, 60, 0],
          showLine: false,
          pointRadius: 5,
          backgroundColor: 'rgb(255, 0, 0)',
        }]
      }
    });

    // Save to jQuery object

    $chart.data('chart', salesChart);

  };


  // Events

  if ($chart.length) {
    init($chart);
  }

})();


//
// Sales chart
//
var SalesChart = (function() {
  // Variables
  var $chart = $('#chart-sales-dark');

  // Methods
  function fetchData() {
      fetch('/run/')
          .then(response => {
              if (!response.ok) {
                  throw new Error('Network response was not ok');
              }
              return response.json();
          })
          .then(data => {
              // Assuming the data is an array of numbers
              initChart($chart, data);
          })
          .catch(error => {
              console.error('Error fetching data:', error);
              alert('Failed to fetch data for the chart: ' + error.message);
          });
  }

  function initChart($chart, chartData) {
      var salesChart = new Chart($chart, {
          type: 'line',
          options: {
              scales: {
                  yAxes: [{
                      gridLines: {
                          lineWidth: 1,
                          color: Charts.colors.gray[900],
                          zeroLineColor: Charts.colors.gray[900]
                      },
                      ticks: {
                          callback: function(value) {
                              if (!(value % 10)) {
                                  return '$' + value + 'k';
                              }
                          }
                      }
                  }]
              },
              tooltips: {
                  callbacks: {
                      label: function(item, data) {
                          var label = data.datasets[item.datasetIndex].label || '';
                          var yLabel = item.yLabel;
                          var content = '';

                          if (data.datasets.length > 1) {
                              content += '<span class="popover-body-label mr-auto">' + label + '</span>';
                          }

                          content += '<span class="popover-body-value">$' + yLabel + 'k</span>';
                          return content;
                      }
                  }
              }
          },
          data: {
              labels: ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
              datasets: [{
                  label: 'Performance',
                  data: chartData, // Use fetched data here
                  showLine: false,
                  pointRadius: 5,
                  backgroundColor: 'rgb(255, 0, 0)',
              }]
          }
      });

      // Save to jQuery object
      $chart.data('chart', salesChart);
  }

  // Fetch data and initialize chart
  fetchData();

})();

