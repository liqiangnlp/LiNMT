my $line_no = 0;
while (my $line = <STDIN>)
{
  ++$line_no;
  $line =~ s/[\r\n]//g;
  $line =~ s/\s+/ /g;
  $line =~ s/"(.+?)">\s/"$1">/g;
  $line =~ s/\s<\//<\//g;
  $line =~ s/<ENAMEX TYPE="(.+?)">(.+?)<\/ENAMEX>/<$1>$2<\/$1>/g;
  print $line."\n";


  if($line_no % 100 == 0)
  {
    print STDERR "\r$line_no";
  }
}
print STDERR "\r$line_no\n";


