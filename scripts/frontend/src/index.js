import React from "react";
import { render } from 'react-dom';
import { ChakraProvider } from "@chakra-ui/react";

import Header from "./components/Header";
import Predictions from "./components/Predictions";

function App() {
  return (
    <ChakraProvider>
      <Header />
      <Predictions />
    </ChakraProvider>
  )
}

const rootElement = document.getElementById("root")
render(<App />, rootElement)