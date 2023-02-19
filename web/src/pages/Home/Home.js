import logo from './logo.svg';
import './Home.css';
import SearchBox from '../../components/SearchBox';

function Home() {
	return (
		<div className="app__home">
			<header className="app__header">
				<img src={logo} className="app__logo" alt="logo" />
			</header>

			<div className="app__content">
				<SearchBox />
			</div>
		</div>
	);
}

export default Home;