import $ from 'jquery'
import Chart from 'chart.js'

/**
 * Functions to generate chartJs data for model
 */

function makeLineChartData (title, xAxisLabel, yAxisLabel) {
  return {
    type: 'scatter',
    data: {datasets: []},
    options: {
      title: {
        display: true,
        text: title,
      },
      legend: {
        display: false,
      },
      maintainAspectRatio: false,
      responsive: true,
      scales: {
        xAxes: [{
          type: 'linear',
          position: 'bottom',
          scaleLabel: {
            display: true,
            labelString: xAxisLabel
          },
          ticks: {}
        }],
        yAxes: [{
          type: 'linear',
          scaleLabel: {
            display: true,
            labelString: yAxisLabel
          }
        }]
      }
    }
  }
}

const colors = [
  '#4ABDAC', // fresh
  '#FC4A1A', // vermilion
  '#F78733', // sunshine
  '#037584', // starry night
  '#007849', // iris
  '#FAA43A', // orange
  '#60BD68', // green
  '#F17CB0', // pink
  '#B2912F', // brown
  '#B276B2', // purple
  '#DECF3F', // yellow
  '#F15854', // red
  '#C08283', // pale gold
  '#dcd0c0', // silk
  '#E37222' // tangerine
]

let seenNames = []

function getColor (name) {
  let i = seenNames.indexOf(name)
  if (i < 0) {
    seenNames.push(name)
    i = seenNames.length - 1
  }
  return colors[i % colors.length]
}

function addDataset (datasets, name, xValues, yValues) {
  let datasetData = []
  for (var i = 0; i < xValues.length; i += 1) {
    datasetData.push({x: xValues[i], y: yValues[i]})
  }
  let dataset = {
    label: name,
    data: datasetData,
    fill: false,
    backgroundColor: getColor(name),
    borderColor: getColor(name),
    showLine: false,
    // pointStyle: 'dash'
  }
  datasets.push(dataset)
}

class ChartsContainer {

  constructor (divTag) {
    this.divTag = divTag
    this.div = $(divTag)
    this.containers = []
  }

  createChart (chartData) {
    let canvas = $('<canvas>')
    let div = $('<div>')
      .addClass('md-card')
      .css({
        'padding': '15px',
        'margin-right': '15px',
        'margin-bottom': '15px',
        'float': 'left',
        'width': '550px',
        'height': '320px'
      })
      .append(canvas)
    let chart = new Chart(canvas, chartData)
    this.containers.push({div, chartData, chart})
    this.div.append(div)
  }

  update (chartDataList) {
    for (let iChart = 0; iChart < chartDataList.length; iChart += 1) {
      if (iChart >= this.containers.length) {
        this.createChart(chartDataList[iChart])
      } else {
        let datasets = this.containers[iChart].chartData.data.datasets
        let newDatasets = chartDataList[iChart].data.datasets
        for (let iDataset = 0; iDataset < newDatasets.length; iDataset += 1) {
          datasets[iDataset].data = newDatasets[iDataset].data
        }
        this.containers[iChart].chart.update()
      }
    }
  }
}

export default { makeLineChartData, addDataset, ChartsContainer }