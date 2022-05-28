import * as React from 'react';
import Grid from '@mui/material/Grid';
import UserForm from './UserForm';
import axios from 'axios';
import ImageWindow from './ImageWindow';
import { saveAs } from 'file-saver'
import Button from '@mui/material/Button';
import { padding } from '@mui/system';
import Container from '@mui/material/Container';

const UPLOAD_URL = 'http://localhost:5000/upload';

export default class Inference extends React.Component{
    constructor(props){
        super(props);
        this.state = {
            uploadStatus: "not_uploaded",
            renderCounter: 0,
            index: 0,
        };
        this.handleSubmit = this.handleSubmit.bind(this);
        this.nextImage = this.nextImage.bind(this);
        this.prevImage = this.prevImage.bind(this);
        this.download = this.download.bind(this);
    }

    handleSubmit(imagew, imageh, uploadFiles){
        const formData = new FormData();
        for (let i = 0; i < uploadFiles.files.length; i++){
            formData.append("file", uploadFiles.files[i]);
        }
        formData.append("imagew", imagew);
        formData.append("imageh", imageh);

        this.setState((state, props) => ({
            ...this.state,
            uploadStatus: "processing"
        }));

        axios.post(UPLOAD_URL, formData).then((res) => {
            this.setState((state, props) => ({
                ...this.state,
                uploadStatus: "done"
            }));
            console.log("file upload success")
        }).catch((err) => alert(err));
    }

    nextImage(){
        axios.get("http://localhost:5000/next_image").then((res) => {
            console.log(res)
            if (res.data !== "error"){
                console.log("next_image");
                const ix = this.state.index + 1
                this.setState((state, props) => ({
                    ...this.state,
                    index: ix
                }));         
            }else{
                alert("no next image!")
            }
        }).catch((err) => alert("Request Error"));
    }

    prevImage(){
        axios.get("http://localhost:5000/prev_image").then((res) => {
            if (res.data !== "error"){
                console.log("prev_image");
                const ix = this.state.index - 1
                this.setState((state, props) => ({
                    ...this.state,
                    index: ix
                }));
            }else{
                alert("no previous image!")
            }
        }).catch((err) => alert("Request Error"));
    }

    download(){
        saveAs("http://localhost:5000/download", 'results.zip')
    }

    render(){
        return(
            <Grid container spacing={2}>
            <Grid item xs={4}>
                <UserForm 
                    submitCallback={(imagew, imageh, files) => this.handleSubmit(imagew, imageh, files)}
                    statusUpdateCallback={(param) => this.setState({uploadStatus: param})}
                />
                <DownloadButton status = {this.state.uploadStatus} onClick = {()=>this.download()}/>
            </Grid>
            <Grid item xs={8}>
                <ImageWindow uploadStatus={this.state.uploadStatus} index={this.state.index} prevImage={this.prevImage} nextImage={this.nextImage}/>
            </Grid>
            </Grid>
            
        );
    }
    
}


function DownloadButton(props){
    console.log("button: "+ props.status)
    if(props.status === "done"){
        return(
            <Container maxWidth="xs">
            <Button 
                variant="outlined" 
                sx={{ mt: 3, mb: 2}}
                onClick = {props.onClick}
            > 
            Download
            </Button>
            </Container>
        );
    }else{
        return(<p></p>);
    }
}