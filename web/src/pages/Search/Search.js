import logo from './logo.svg';
import './Search.css';
import SearchBox from '../../components/SearchBox';
import Loader from '../../components/Loader';
import { useParams, Link } from "react-router-dom";
import { useEffect, useRef, useState } from 'react';
import Paper from "@material-ui/core/Paper";
import SearchIcon from '@material-ui/icons/Search';
import Plot from 'react-plotly.js';
import { useLocation } from "react-router-dom";
import { BASE_API_URI } from "../../services/api";

function NoOccurences() {
	return (
		<Paper style={{ backgroundColor: "unset", border: "1px solid #5f6368", borderRadius: "25px", boxShadow: "unset", padding: "0 20px 0.83em 20px" }}>
			<div className="no-result">
				<div className="no-result__content">
					<h2>Sembra che non ci sia nessuna corrispondenza per la tua ricerca</h2>
					<p>Suggerimenti per la ricerca:</p>
					<ul>
						<li>Assicurati che tutte le parole siano state gitiate correttamente</li>
						<li>Prova ad utilizzare i tag!</li>
						<li>Prova con meno parole</li>
					</ul>
				</div>
			</div>
		</Paper>
	);
}

function Search() {
	const params = useParams();

	const search = useLocation().search;
	const query = new URLSearchParams(search).get('q') || "";

	const [error, setError] = useState(null);
	const [items, setItems] = useState([]);
	const [isLoaded, setIsLoaded] = useState(false);
	const [itemWidth, setItemWidth] = useState(null);

	useEffect(() => {
		if (!query)
			return;

		setIsLoaded(false);
		fetch(`${BASE_API_URI}/api/search-by-query/${query}`)
			.then(res => res.json())
			.then(
				(result) => {
					setIsLoaded(true);
					setItems(result.similarImages);
				},
				(error) => {
					setIsLoaded(true);
					setError(error);
				}
			)
	}, [query])

	useEffect(() => {
		if (!params.value)
			return;

		setIsLoaded(false);
		fetch(`${BASE_API_URI}/api/search-by-object/${params.value}`)
			.then(res => res.json())
			.then(
				(result) => {
					setIsLoaded(true);
					setItems(result.similarImages);
				},
				(error) => {
					setIsLoaded(true);
					setError(error);
				}
			)
	}, [params.value])

	const hasOccurences = items && items.length > 0;
	const resultRef = useRef(null);

	const handleResize = function() {
		const minWidth = 200;

		if (resultRef && resultRef.current) {
			const availableWidth = resultRef.current.offsetWidth;
			const itemsPerRow = Math.min(7, Math.floor(availableWidth / minWidth));
			const width = Math.floor((availableWidth - 10 * itemsPerRow) / itemsPerRow);
			setItemWidth(width);
		}
	};

	useEffect(handleResize, [resultRef]);

	useEffect(() => {
		handleResize();
		window.addEventListener('resize', handleResize);
		return _ => window.removeEventListener('resize', handleResize);
	});

	return (
		<div className="app__search">
			<header className="app__header">
				<Link to="/">
					<img src={logo} className="app__logo" alt="logo" />
				</Link>

				<SearchBox value={query} />
			</header>

			<div className="app__content">
				{!isLoaded && <Loader />}
				{isLoaded && !hasOccurences && <NoOccurences />}
				{isLoaded && hasOccurences && (
					<div className="search-result" ref={resultRef}>
						{itemWidth && itemWidth > 0 && items.map((item, idx) =>
							<div className="search-item" key={idx}>
								<Plot
									layout={{
										width: itemWidth,
										height: itemWidth,

										margin: {
											l: 10,
											r: 10,
											b: 10,
											t: 10
										},
									}}

									data={[
										{
											x: item.image.map(row => row[0]),
											y: item.image.map(row => row[2]),
											z: item.image.map(row => row[1]),
											type: 'scatter3d',
											mode: 'markers',
											marker: {
												color: 'b',
												size: 1.5,
												opacity: 0.8,
											},
										},
									]}
								/>
							</div>
						)}
					</div>
				)}
			</div>
		</div>
	);
}

export default Search;