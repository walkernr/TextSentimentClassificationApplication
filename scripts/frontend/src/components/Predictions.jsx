import React, { useEffect, useState } from "react";
import {Box, Button, Flex, Input, InputGroup,
        Modal, ModalBody, ModalCloseButton, ModalContent, ModalFooter, ModalHeader, ModalOverlay,
        Stack, Text, useDisclosure} from "@chakra-ui/react";

const PredictionsContext = React.createContext({predictions: [], fetchPredictions: () => {}})

function AddPrediction() {
    const [item, setItem] = React.useState("")
    const {predictions, fetchPredictions} = React.useContext(PredictionsContext)
    const handleInput = event  => {setItem(event.target.value)}
    const handleSubmit = () => {
        const newPrediction = {"id": predictions.length,"item": item}
        fetch("http://localhost:8000/prediction", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(newPrediction)
        }).then(fetchPredictions)
    }
    return (
        <form onSubmit={handleSubmit}>
            <InputGroup size="md">
                <Input
                    pr="4.5rem"
                    type="text"
                    placeholder="Add a prediction item"
                    aria-label="Add a prediction item"
                    onChange={handleInput}
                />
            </InputGroup>
        </form>
    )
}

export default function Predictions() {
    const [predictions, setPredictions] = useState([])
    const fetchPredictions = async () => {
        const response = await fetch("http://localhost:8000/prediction")
        const predictions = await response.json()
        setPredictions(predictions.data)
    }
    useEffect(() => {fetchPredictions()}, [])
    return (
        <PredictionsContext.Provider value={{predictions, fetchPredictions}}>
            <AddPrediction />
            <Stack spacing={5}>
                {predictions.map((prediction) => (<b>{prediction.item}</b>))}
            </Stack>
        </PredictionsContext.Provider>
    )
}