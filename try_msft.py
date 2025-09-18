from main import build_symbol, list_symbols

if __name__ == '__main__':
    print('Attempting to build MSFT...')
    r = build_symbol('MSFT')
    print('build_symbol:', r)
    print('Listing symbols now:')
    print(list_symbols())
