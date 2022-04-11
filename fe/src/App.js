import logo from './logo.svg';
import './App.css';

import ReactSearchBox from "react-search-box";


import GraphViz from './GraphViz'


function App() {
  const data = [
    {
      key: "john",
      value: "John Doe",
    },
    {
      key: "jane",
      value: "Jane Doe",
    },
    {
      key: "mary",
      value: "Mary Phillips",
    },
    {
      key: "robert",
      value: "Robert",
    },
    {
      key: "karius",
      value: "Karius",
    },
  ]

  return ([
    <ReactSearchBox
       placeholder="Placeholder"
       value="Doe"
       data={data}
       callback={(record) => console.log(record)}
     />,
    <GraphViz/>
  ]
  );
}

export default App;
