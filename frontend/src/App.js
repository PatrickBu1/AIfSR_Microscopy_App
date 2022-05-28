import * as React from 'react';
import AppBar from '@mui/material/AppBar';
import Button from '@mui/material/Button';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import { green, grey } from '@mui/material/colors';
import vipLogo from './assets/vip_logo.png'
import Inference from './Inference';
import { fontFamily } from '@mui/system';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';

export const baseURL = "localhost:5000/";

const theme = createTheme({
  palette: {
    primary: {
      main: '#57068c',
    },
    secondary: {
      main:'#DEDEDE'
    }
  },
  typography: {
    fontSize: 16,
    fontFamily: 'Arial'
  },
  spacing: 4
});

export default class App extends React.Component{
  constructor(props){
    super(props);
    this.state = {
      imageLoaded: false,
      currentPage: 0
    };
    // this binding to event handlers
    this.aifsrRedirect = this.aifsrRedirect.bind(this);
    this.pageRedirect = this.pageRedirect.bind(this);
  }

  aifsrRedirect(){
    const newUrl = "https://engineering.nyu.edu/research-innovation/student-research/vertically-integrated-projects/vip-teams/ai-scientific-research";
    window.location.href = newUrl;
  }

  pageRedirect(page){
    this.setState({currentPage: page});
  }

  appBar = () => (
    <AppBar position="static">
      <Toolbar>
        <img src={vipLogo} alt="Logo" width={70} height={70} />
        <Typography variant="h6" color="inherit" sx={{ margin: 4,  flexGrow: 1, fontFamily: 'Arial'}} noWrap>
          NYU AIfSR Microscopy Team
        </Typography>
        <Button color="inherit" onClick={() => this.pageRedirect(0)} sx={{margin: 2, fontFamily: 'Arial'}}>Demo Page</Button>
        <Button color="inherit" onClick={() => this.pageRedirect(1)} sx={{margin: 2, fontFamily: 'Arial'}}>Team Members</Button>
        <Button color="inherit" onClick={this.aifsrRedirect}>Model Used</Button>
      </Toolbar>
    </AppBar>
  );

  render(){
    if (this.state.currentPage === 0){
      return(
        <ThemeProvider theme={theme}>
        <this.appBar/>
        <Box sx={{marginTop: 5, display: 'flex', flexDirection: 'column', alignItems: 'center',}}>
          <Typography mx={{margin: 30, fontFamily: "Arial", fontSize: 30}}>
            Semantic Segmentation - Sperm Image Demo
          </Typography>
        </Box>
        <Inference/>
        </ThemeProvider>
         
      );
    }
    else{
      return(
        <ThemeProvider theme={theme}>
        <this.appBar/>
        <h1>THIS IS THE TEAM PAGE</h1>
        </ThemeProvider>
      );
    }
    
  }
}