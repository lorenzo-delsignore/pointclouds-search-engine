import React from "react";
import PropTypes from "prop-types";
import IconButton from "@material-ui/core/IconButton";
import Input from "@material-ui/core/Input";
import Paper from "@material-ui/core/Paper";
import ClearIcon from "@material-ui/icons/Clear";
import SearchIcon from "@material-ui/icons/Search";
import InsertPhotoIcon from '@material-ui/icons/InsertPhoto';
import withStyles from "@material-ui/core/styles/withStyles";
import classNames from "classnames";
import { DropzoneArea } from 'material-ui-dropzone';
import { MuiThemeProvider, createTheme } from "@material-ui/core/styles";


const styles = (theme) => ({
  rootContainer: {
    position: "relative",
    zIndex: "100",
  },
  root: {
    display: "flex",
    boxShadow: "unset",
    backgroundColor: "unset",
    height: theme.spacing(6),
    justifyContent: "space-between",
    border: "1px solid #5f6368",
    borderRadius: "24px",
    padding: "0 5px"
  },
  dragZoneArea: {
    position: "absolute",
    right: 0,
    left: 0,
    top: 0,
  },
  iconButton: {
    // color: theme.palette.action.active,
    color: "#e8eaed",
    transform: "scale(1, 1)",
    transition: theme.transitions.create(["transform", "color"], {
      duration: theme.transitions.duration.shorter,
      easing: theme.transitions.easing.easeInOut,
    }),
  },
  iconButtonHidden: {
    transform: "scale(0, 0)",
    "& > $icon": {
      opacity: 0,
    },
  },
  searchIconButton: {
    marginRight: theme.spacing(-6),
  },
  icon: {
    transition: theme.transitions.create(["opacity"], {
      duration: theme.transitions.duration.shorter,
      easing: theme.transitions.easing.easeInOut,
    }),
  },
  input: {
    width: "100%",
    color: "#e8eaed",
    // padding: "0 5px",
  },
  searchContainer: {
    margin: "auto 16px",
    width: `calc(100% - ${theme.spacing(6 + 4)}px)`, // 6 button + 4 margin
  },
  closeDropzoneAreaIcon: {
    position: "absolute",
    zIndex: "101",
    right: "5px",
    top: "5px",
  }
});


const theme = createTheme({
  overrides: {
    MuiDropzoneArea: {
      root: {
        color: "#e8eaed",
        borderRadius: "25px",
        backgroundColor: "#282c34",
        borderColor: "#5f6368",
        border: "1px solid",
        minHeight: "250px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: "100"
      },
      icon: {
        color: "#e8eaed",
      }
    },
  }
});

/**
 * Material design search bar
 * @see [Search patterns](https://material.io/archive/guidelines/patterns/search.html)
 */
const SearchBar = React.forwardRef(
  (
    {
      cancelOnEscape,
      className,
      classes,
      closeIcon,
      disabled,
      onCancelSearch,
      onRequestSearch,
      insertPhotoIcon,
      onDragFile,
      searchIcon,
      style,
      ...inputProps
    },
    ref
  ) => {
    const inputRef = React.useRef();
    const [value, setValue] = React.useState(inputProps.value);
    const [showDragzone, showInsertImageDragzone] = React.useState(false);

    React.useEffect(() => {
      setValue(inputProps.value);
    }, [inputProps.value]);

    const handleFocus = React.useCallback(
      (e) => {
        if (inputProps.onFocus) {
          inputProps.onFocus(e);
        }
      },
      [inputProps.onFocus]
    );

    const handleBlur = React.useCallback(
      (e) => {
        setValue((v) => v.trim());
        if (inputProps.onBlur) {
          inputProps.onBlur(e);
        }
      },
      [inputProps.onBlur]
    );

    const handleInput = React.useCallback(
      (e) => {
        setValue(e.target.value);
        if (inputProps.onChange) {
          inputProps.onChange(e.target.value);
        }
      },
      [inputProps.onChange]
    );

    const handleCancel = React.useCallback(() => {
      setValue("");
      if (onCancelSearch) {
        onCancelSearch();
      }
    }, [onCancelSearch]);

    const handleRequestSearch = React.useCallback(() => {
      if (onRequestSearch) {
        onRequestSearch(value);
      }
    }, [onRequestSearch, value]);

    const handleDragFile = (files) => {
      showInsertImageDragzone(false);

      if (onDragFile) {
        onDragFile(files);
      }
    };

    const handleKeyUp = React.useCallback(
      (e) => {
        if (e.charCode === 13 || e.key === "Enter") {
          handleRequestSearch();
        } else if (
          cancelOnEscape &&
          (e.charCode === 27 || e.key === "Escape")
        ) {
          handleCancel();
        }
        if (inputProps.onKeyUp) {
          inputProps.onKeyUp(e);
        }
      },
      [handleRequestSearch, cancelOnEscape, handleCancel, inputProps.onKeyUp]
    );

    React.useImperativeHandle(ref, () => ({
      focus: () => {
        inputRef.current.focus();
      },
      blur: () => {
        inputRef.current.blur();
      },
    }));

    return (
      <div className={classNames(classes.rootContainer)}>
        <Paper className={classNames(classes.root, className)} style={style} onDragOver={() => showInsertImageDragzone(true)}>
          <div className={classes.searchContainer}>
            <Input
              {...inputProps}
              inputRef={inputRef}
              onBlur={handleBlur}
              value={value}
              onChange={handleInput}
              onKeyUp={handleKeyUp}
              onFocus={handleFocus}
              fullWidth
              className={classes.input}
              disableUnderline
              disabled={disabled}
            />
          </div>
          <IconButton
            onClick={handleRequestSearch}
            className={classNames(classes.iconButton, classes.searchIconButton, {
              [classes.iconButtonHidden]: value !== "",
            })}
            disabled={disabled}
          >
            {React.cloneElement(searchIcon, {
              classes: { root: classes.icon },
            })}
          </IconButton>
          <IconButton
            onClick={handleCancel}
            className={classNames(classes.iconButton, {
              [classes.iconButtonHidden]: value === "",
            })}
            disabled={disabled}
          >
            {React.cloneElement(closeIcon, {
              classes: { root: classes.icon },
            })}
          </IconButton>
          <IconButton
            className={classNames(classes.iconButton, className)}
            onClick={() => showInsertImageDragzone(true)}
          >
            {React.cloneElement(insertPhotoIcon, {
              classes: { root: classes.icon },
            })}
          </IconButton>
        </Paper>

        {showDragzone &&
          <div className={classNames(classes.dragZoneArea, className)} style={style}>
            <MuiThemeProvider theme={theme}>
              <DropzoneArea
                acceptedFiles={['.obj']}
                dropzoneText={"Trascina un'immagine qui o carica un file"}
                onDrop={handleDragFile}
                maxFileSize={100000000}
                filesLimit={1}
                onAlert={null}
              />
            </MuiThemeProvider>
            <IconButton
              onClick={() => showInsertImageDragzone(false)}
              className={classNames(classes.iconButton, classes.closeDropzoneAreaIcon)}
            >
              {React.cloneElement(closeIcon, {
                classes: { root: classes.icon },
              })}
            </IconButton>
          </div>
        }
      </div>
    );
  }
);

SearchBar.defaultProps = {
  className: "",
  closeIcon: <ClearIcon />,
  disabled: false,
  placeholder: "Search",
  searchIcon: <SearchIcon />,
  insertPhotoIcon: <InsertPhotoIcon />,
  style: null,
  value: "",
};

SearchBar.propTypes = {
  /** Whether to clear search on escape */
  cancelOnEscape: PropTypes.bool,
  /** Override or extend the styles applied to the component. */
  classes: PropTypes.object.isRequired,
  /** Custom top-level class */
  className: PropTypes.string,
  /** Override the close icon. */
  closeIcon: PropTypes.node,
  /** Disables text field. */
  disabled: PropTypes.bool,
  /** Fired when the search is cancelled. */
  onCancelSearch: PropTypes.func,
  /** Fired when the text value changes. */
  onChange: PropTypes.func,
  /** Fired when the search icon is clicked. */
  onRequestSearch: PropTypes.func,
  /** Fired when the search icon is clicked. */
  onDragFile: PropTypes.func,
  /** Sets placeholder text for the embedded text field. */
  placeholder: PropTypes.string,
  /** Override the search icon. */
  searchIcon: PropTypes.node,
  /** Override the insert photo icon. */
  insertPhotoIcon: PropTypes.node,
  /** Override the inline-styles of the root element. */
  style: PropTypes.object,
  /** The value of the text field. */
  value: PropTypes.string,
};

export default withStyles(styles)(SearchBar);