import React, { Component } from 'react';

import { ForceGraph3D  } from 'react-force-graph'

// <div onload="getStories();"></div>
export class GraphViz extends Component {

  constructor(props) {
    super(props);
    // Don't call this.setState() here!
    this.state = { gData: '' };
    // this.handleClick = this.handleClick.bind(this);
  }
  shouldComponentUpdate(nextProps, nextState) {
      const shouldUpdate = nextState.gData != this.state.gData
      console.log('shouldUpdate:', shouldUpdate);
      return shouldUpdate
  }

  componentDidMount () {
    fetch('http://localhost:8000/sample_data')
      .then(res => res.json())
      .then(d => {
          console.log('respodned_data', d)
          this.setState({gData: d})
      })
  }


  render() {

  const props = this.props
  const gData = this.state.gData

  if (gData === '') {
    return 'loading'
  }

  return (
      <div>
      graph viz baby!

      <ForceGraph3D
      graphData={gData} />

      </div>
    )
  }
}


export default GraphViz;
