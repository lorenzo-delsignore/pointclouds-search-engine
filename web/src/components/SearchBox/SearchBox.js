import SearchBar from "../SearchBar";
import { DropzoneArea } from 'material-ui-dropzone';
import { MuiThemeProvider, createTheme } from "@material-ui/core/styles";
import { useNavigate } from "react-router-dom";
import { BASE_API_URI } from "../../services/api";

const theme = createTheme({
    overrides: {
        MuiDropzoneArea: {
            root: {
                color: "#e8eaed",
                borderRadius: "25px",
                backgroundColor: "unset",
                borderColor: "#5f6368",
                border: "1px solid",
                minHeight: "250px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center"
            },
            icon: {
                color: "#e8eaed",
            }
        },
        MuiDropzoneSnackbar: {
            errorAlert: {
                backgroundColor: "#AFA",
                color: "#000"
            },
            successAlert: {
                backgroundColor: "#FAA",
                color: "#000"
            },
        },
    }
});

function SearchBox({ value }) {
    const navigate = useNavigate();
    const onRequestSearch = (value) => {
        if (!value)
            return;
            
        navigate(`/search?q=${value}`);
    };

    const onDragFile = (files) => {
        if (!files || files.length !== 1)
            return false;

        var data = new FormData()
        data.append('file', files[0])

        fetch(`${BASE_API_URI}/api/upload`, {
            method: 'POST',
            body: data,
        })
        .then(res => res.json())
        .then(
            (result) => {
                if (result && result.id) {
                    navigate(`/search/${result.id}`);
                }
            },
            (error) => {
                console.log(error);
            }
        )
    };

    return (
        <div className="search-box">
            <SearchBar 
                value={value || ""}
                onDragFile={onDragFile}
                onRequestSearch={onRequestSearch}
            />
        </div>
    );
}

export default SearchBox;