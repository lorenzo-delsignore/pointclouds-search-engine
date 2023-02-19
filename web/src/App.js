// utilizzare le rotte, se / mostra homepage, altrimenti la search page /search

import './App.css';
import Home from "./pages/Home/Home";
import Search from "./pages/Search/Search";

import {
	Routes,
	Route,
	Link
} from "react-router-dom";

function App() {
	return (
		<div className="app">
			<Routes>
				<Route path="/search/:value" element={<Search />} />
				<Route path="/search" element={<Search />} />
				<Route exact path="/" element={<Home />} />
			</Routes>
		</div>
	);
}

export default App;