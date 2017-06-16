my @labels = qw($day $literal $url $person $location $corporation $reserved-1 $reserved-2 $reserved-3 $reserved-4 $reserved-5 $reserved-6 $reserved-7 $reserved-8 $reserved-9 $reserved-10);

foreach my $word (@labels)
{
  foreach(1..10)
  {
    my $line = "";
    foreach(1..30)
    {
      $line .= " $word";
    } 
    $line =~ s/\s+$//g;
    $line =~ s/^\s+//g;
    print $line."\n";
  }
}
