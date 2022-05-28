import * as React from 'react';
import Grid from '@mui/material/Grid';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import { Container, Typography, Button} from '@mui/material';
import {useImage} from 'react-image'

export default function ImageWindow(props){
    return(
        <Container maxWidth='lg' fixed>
            <Box 
                width={'100%'}
                height={500}
                display="flex" 
                alignItems="center"
                justifyContent="center"
                sx={{backgroundColor: 'secondary.main',
                    borderRadius: 3,
                    border: 1,
                    borderStyle: 'solid'
                }}>
                <CenterItem  index={props.index} uploadStatus={props.uploadStatus} prevImage={props.prevImage} nextImage={props.nextImage}/>
                
            </Box>
        </Container>
    );    
}


function CenterItem(props){
    if (props.uploadStatus === "not_uploaded"){
        return(
            <Typography>output image preview will be shown here after uploading.</Typography>
        );
    }else if (props.uploadStatus === "processing"){
        return(
            <CircularProgress color="primary" />
        );
    }else{
        const imgSrc = 'http://localhost:5000/get_image/' + (props.index).toString()
        const maskSrc = 'http://localhost:5000/get_mask/' + (props.index).toString()
        return(
            <Container maxWidth='xl'>
            <Grid container spacing={2}>
                <Grid item xs={6}>
                    <img src={imgSrc} style={{margin: 10}}width={450} height={400}></img>
                </Grid>
                <Grid item xs={6}>
                    <img src={maskSrc} style={{margin: 10}} width={450} height={400} ></img>
                </Grid>
            </Grid>

            <Grid container spacing={2}>
                <Grid item xs={6}>
                    <Button variant="outlined" onClick={props.prevImage} fullWidth>Previous Image</Button>
                </Grid>
                <Grid item xs={6}>
                    <Button variant="outlined" onClick={props.nextImage} fullWidth>Next Image</Button>
                </Grid>
            </Grid>
            </Container>
            
        )
    }
}