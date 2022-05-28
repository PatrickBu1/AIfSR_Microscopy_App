import * as React from 'react';
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import Container from '@mui/material/Container';

export default class UserForm extends React.Component{
    constructor(props){
        super(props);
        this.state = {
            imagew: '',
            imageh: '',
            files: null,
            error: '',
        };
        this.handleInputChange = this.handleInputChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
        //this.handleFileInput = this.handleFileInput.bind(this);
        this.submitCallback = props.submitCallback;
        this.uploadInput = null;
        this.statusUpdateCallback = props.statusUpdateCallback;
    }

    handleInputChange(event){
        const target = event.target;
        this.setState({
            [target.name]: target.value
        });
    }

    
    // handleFileInput(event){
    //     this.setState({files: event.target.files});
    // }

    handleSubmit(event){
        event.preventDefault();
        if (isNaN(parseInt(this.state.imagew, 10)) || isNaN(parseInt(this.state.imageh, 10)) || this.uploadInput == null){
            alert('Inputs are not valid!')
        }
        else{
            // this.statusUpdateCallback("processing");
            this.submitCallback(this.state.imagew, this.state.imageh, this.uploadInput);
            //this.statusUpdateCallback("done");
        }
    }

    render(){
        return(
            <Container component="main" maxWidth="xs">
                <form onSubmit={this.handleSubmit}>
                    <TextField 
                        margin="normal"
                        value={this.state.imagew} 
                        id="imagew" 
                        label="output image width" 
                        name="imagew" 
                        onChange = {this.handleInputChange}
                        fullWidth/>
                    <TextField 
                        margin="normal" 
                        value={this.state.imageh} 
                        name="imageh" 
                        label="output image height" 
                        id="imageh" 
                        onChange = {this.handleInputChange}
                        fullWidth/>
                    <input type="file" ref={(ref)=> {this.uploadInput = ref;}} multiple/>
                    <Button 
                        type="submit" 
                        variant="contained" 
                        sx={{ mt: 3, mb: 2 }}
                    > 
                    Upload and process
                    </Button>
                </form>
            </Container>
        );
    }
    
}
